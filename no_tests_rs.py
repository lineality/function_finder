"""
no_tests_rs module
==================

Produces "tests removed" copies of Rust source files.

For every input `.rs` file, this module writes a parallel output file
identical to the original EXCEPT that cargo-test code is stripped out:
  - `#[cfg(test)] mod tests { ... }` blocks
  - Standalone `#[test] fn ...` items (and `#[tokio::test]`,
    `#[async_std::test]`)

Function bodies are NOT removed (that is a separate concern handled by
`functional_table_of_contents.py`). Everything else in the source --
doc-comments, attributes, structs, enums, traits, impls, type aliases,
constants, statics, uses, mods, function bodies of non-test functions --
is preserved verbatim.

The original source file is NEVER modified. Each output is written to a
separate file (default: `no_tests_files_<timestamp>/no_tests.<name>.rs`).

Public API
----------
    no_tests_rs(...)
        Process a single `.rs` file or a directory of `.rs` files.

Design notes / known limitations
--------------------------------
- Line-based heuristic, NOT a Rust AST parser. Mirrors the style of
  `function_finder.py` and `functional_table_of_contents.py`.
- Brace counting (used to find the end of an `#[cfg(test)] mod`) is
  naive. Braces inside string literals, character literals, raw strings,
  comments, and `macro_rules!` bodies may confuse the counter. Accepted
  per spec.
- Multi-line attributes (`#[derive(\n    Foo,\n)]`) are only partially
  handled: each line starting with `#[` is treated as an attribute line.

Python version: 3.9+ (uses PEP 604 `str | None` style unions).
"""

import os
import re
import time
import traceback
from datetime import datetime


# ============================================================================
# Module-level constants
# ============================================================================

# Default filename prefix for the produced output files.
# Example: source "foo.rs" -> output "no_tests.foo.rs".
DEFAULT_NO_TESTS_FILENAME_PREFIX: str = "no_tests."

# Default subdirectory base name (created inside cwd or `output_dir`)
# into which output files are written. The actual subdirectory name is
# this string suffixed with a run timestamp.
DEFAULT_OUTPUT_SUBDIRECTORY_NAME: str = "no_tests_files"

# Attribute substrings that mark a Rust item as test-only. Any line
# whose stripped form CONTAINS one of these substrings (and starts with
# `#[`) is treated as the start (or part) of a test region to strip.
TEST_MARKER_ATTRIBUTE_SUBSTRINGS_TUPLE: tuple[str, ...] = (
    "#[cfg(test)]",
    "#[test]",
    "#[tokio::test]",
    "#[async_std::test]",
)


# ============================================================================
# Generic helpers (style mirrors function_finder.py)
# ============================================================================

def _generate_iso8601_timestamp_string() -> str:
    """
    Generate a filesystem-safe ISO8601 timestamp string with microsecond
    precision (colons replaced by underscores).

    Returns
    -------
    str
        e.g. '2024-01-15T14_32_45_123456'.
    """
    current_datetime_object = datetime.now()
    raw_iso_string = current_datetime_object.isoformat(timespec="microseconds")
    filesystem_safe_iso_string = raw_iso_string.replace(":", "_")
    return filesystem_safe_iso_string


def _log_error_message_to_file_and_terminal(
    error_log_absolute_file_path: str,
    error_message_text: str,
) -> None:
    """
    Append an error message to the run's error log file AND print it to
    the terminal. Never raises.

    Parameters
    ----------
    error_log_absolute_file_path : str
        Absolute path to the error log file (created/appended).
    error_message_text : str
        Full error message (typically including a traceback) to record.
    """
    # Always print first so the user sees the error even if log write fails.
    print(f"[no_tests_rs ERROR] {error_message_text}")
    try:
        with open(
            error_log_absolute_file_path, "a", encoding="utf-8"
        ) as error_log_file_handle:
            error_log_file_handle.write(error_message_text)
            error_log_file_handle.write("\n" + ("-" * 80) + "\n")
    except Exception as inner_log_write_exception:
        # Swallow secondary failures; we already printed the primary error.
        print(
            "[no_tests_rs ERROR] Additionally, failed to write the above "
            f"error to the log file at {error_log_absolute_file_path!r}: "
            f"{inner_log_write_exception!r}"
        )


def _collect_rust_source_files_with_depth_limit(
    root_directory_absolute_path: str,
    maximum_nested_directory_depth: int,
) -> list[str]:
    """
    Walk a directory tree and return absolute paths of all `.rs` files
    found up to a given nested-directory depth.

    Parameters
    ----------
    root_directory_absolute_path : str
        Directory to search.
    maximum_nested_directory_depth : int
        0 = only files directly inside the root; N = up to N nested levels.

    Returns
    -------
    list[str]
        Sorted list of absolute paths to discovered `.rs` files.

    Raises
    ------
    FileNotFoundError
        If `root_directory_absolute_path` is not an existing directory.
    """
    if not os.path.isdir(root_directory_absolute_path):
        raise FileNotFoundError(
            "Cannot collect Rust files: directory does not exist or is not "
            f"a directory: {root_directory_absolute_path!r}"
        )

    normalized_root_absolute_path = os.path.abspath(root_directory_absolute_path)
    discovered_rust_file_absolute_paths_list: list[str] = []

    for current_walked_directory_path, _subdirectory_names_list, file_names_in_current_dir in os.walk(
        normalized_root_absolute_path
    ):
        relative_path_from_root = os.path.relpath(
            current_walked_directory_path, normalized_root_absolute_path
        )
        if relative_path_from_root == ".":
            current_depth_below_root_integer = 0
        else:
            current_depth_below_root_integer = len(
                relative_path_from_root.split(os.sep)
            )

        if current_depth_below_root_integer > maximum_nested_directory_depth:
            continue

        for single_file_name_string in file_names_in_current_dir:
            if single_file_name_string.endswith(".rs"):
                discovered_rust_file_absolute_paths_list.append(
                    os.path.join(
                        current_walked_directory_path, single_file_name_string
                    )
                )

    discovered_rust_file_absolute_paths_list.sort()
    return discovered_rust_file_absolute_paths_list


# ============================================================================
# Brace counting (intentionally naive, per spec)
# ============================================================================

def _find_matching_close_brace_line_index_via_brace_counting(
    file_text_lines_list: list[str],
    search_start_line_index: int,
) -> int:
    """
    Find the zero-indexed line containing the close brace `}` that
    matches the FIRST open brace `{` found at or after
    `search_start_line_index`, using naive character-by-character brace
    counting.

    Parameters
    ----------
    file_text_lines_list : list[str]
        The full content of a `.rs` file split into lines.
    search_start_line_index : int
        Zero-indexed line at which to begin scanning.

    Returns
    -------
    int
        Zero-indexed line of the matching close brace. If a signature-
        like construct ending with `;` is encountered before any `{`,
        the line index of that `;` is returned (caller interprets as
        "no body"). If no balanced close brace is found, the last line
        index of the file is returned as a safe fallback.

    Limitations
    -----------
    Braces inside string literals, character literals, raw strings, or
    comments will mislead this counter. Accepted per project spec.
    """
    running_brace_balance_integer = 0
    have_seen_first_opening_brace_flag = False
    total_file_line_count_integer = len(file_text_lines_list)

    current_line_index_integer = search_start_line_index
    while current_line_index_integer < total_file_line_count_integer:
        single_line_string = file_text_lines_list[current_line_index_integer]

        for single_character_in_line in single_line_string:
            if single_character_in_line == "{":
                running_brace_balance_integer += 1
                have_seen_first_opening_brace_flag = True
            elif single_character_in_line == "}":
                running_brace_balance_integer -= 1

            if (
                have_seen_first_opening_brace_flag
                and running_brace_balance_integer == 0
            ):
                return current_line_index_integer

        # Signature-only line (e.g. trait method declaration ending in `;`).
        if (
            not have_seen_first_opening_brace_flag
            and single_line_string.rstrip().endswith(";")
        ):
            return current_line_index_integer

        current_line_index_integer += 1

    return total_file_line_count_integer - 1


# ============================================================================
# Test region detection
# ============================================================================

def _line_is_test_marker_attribute(single_line_string: str) -> bool:
    """
    Return True if the given line is (after stripping whitespace) a Rust
    attribute line that marks the following item as test-only.

    Parameters
    ----------
    single_line_string : str
        A single line of Rust source.

    Returns
    -------
    bool
        True if the stripped line starts with `#[` AND contains any of
        the substrings in `TEST_MARKER_ATTRIBUTE_SUBSTRINGS_TUPLE`.
    """
    stripped_line_string = single_line_string.strip()
    if not stripped_line_string.startswith("#["):
        return False
    for single_marker_substring_string in TEST_MARKER_ATTRIBUTE_SUBSTRINGS_TUPLE:
        if single_marker_substring_string in stripped_line_string:
            return True
    return False


def _climb_upward_to_include_preceding_doc_and_attribute_lines(
    file_text_lines_list: list[str],
    anchor_line_index: int,
) -> int:
    """
    Starting from `anchor_line_index`, climb upward through immediately-
    preceding `///`, `//!`, and `#[...]` lines and return the topmost
    line index that still belongs to this item's region.

    Parameters
    ----------
    file_text_lines_list : list[str]
        The full content of a `.rs` file split into lines.
    anchor_line_index : int
        Zero-indexed line to start climbing from.

    Returns
    -------
    int
        Zero-indexed top boundary line (<= anchor_line_index). Blank
        lines and any non-doc/non-attribute line stop the climb.
    """
    top_boundary_line_index_integer = anchor_line_index
    candidate_line_index_integer = anchor_line_index - 1
    while candidate_line_index_integer >= 0:
        candidate_line_stripped_string = file_text_lines_list[
            candidate_line_index_integer
        ].strip()
        is_triple_slash_doc_comment_line = candidate_line_stripped_string.startswith("///")
        is_inner_doc_comment_line = candidate_line_stripped_string.startswith("//!")
        is_attribute_line = candidate_line_stripped_string.startswith("#[")
        if (
            is_triple_slash_doc_comment_line
            or is_inner_doc_comment_line
            or is_attribute_line
        ):
            top_boundary_line_index_integer = candidate_line_index_integer
            candidate_line_index_integer -= 1
        else:
            break
    return top_boundary_line_index_integer


def _find_test_regions_to_strip_from_file(
    file_text_lines_list: list[str],
) -> list[tuple[int, int]]:
    """
    Identify line ranges (inclusive) that comprise test-only code and
    should be removed entirely from the output.

    Strategy
    --------
    1. Scan the file top-to-bottom.
    2. When a line matching `_line_is_test_marker_attribute` is found:
       a. Climb upward through preceding doc-comments and OTHER
          attribute lines to find the true top of the region.
       b. Climb downward through any subsequent doc-comments, blank
          lines, and additional attribute lines until the first
          definition line is reached (the item this attribute applies
          to: typically `mod` or `fn`).
       c. From that definition line, use naive brace counting to find
          the matching close brace (or `;` for bodyless signatures).
       d. Record (top_line, close_line) as a region to strip.
    3. Resume scanning AFTER the stripped region (so nested `#[test]`
       attributes inside an already-stripped `#[cfg(test)] mod` are
       not double-counted).

    Parameters
    ----------
    file_text_lines_list : list[str]
        The full content of a `.rs` file split into lines.

    Returns
    -------
    list[tuple[int, int]]
        Sorted list of (start_line_index, end_line_index) inclusive
        ranges to strip. May be empty.
    """
    test_regions_to_strip_list: list[tuple[int, int]] = []
    total_file_line_count_integer = len(file_text_lines_list)
    current_scan_line_index_integer = 0

    while current_scan_line_index_integer < total_file_line_count_integer:
        single_line_string = file_text_lines_list[current_scan_line_index_integer]

        if not _line_is_test_marker_attribute(single_line_string):
            current_scan_line_index_integer += 1
            continue

        # 2a. Climb upward through preceding doc/attribute lines.
        region_top_line_index_integer = (
            _climb_upward_to_include_preceding_doc_and_attribute_lines(
                file_text_lines_list, current_scan_line_index_integer
            )
        )

        # 2b. Climb downward through additional attribute/doc/blank lines
        # until we land on the actual item definition line.
        definition_line_index_integer = current_scan_line_index_integer + 1
        while definition_line_index_integer < total_file_line_count_integer:
            downward_line_stripped_string = file_text_lines_list[
                definition_line_index_integer
            ].strip()
            is_attribute_line = downward_line_stripped_string.startswith("#[")
            is_doc_comment_line = (
                downward_line_stripped_string.startswith("///")
                or downward_line_stripped_string.startswith("//!")
            )
            is_blank_line = downward_line_stripped_string == ""
            if is_attribute_line or is_doc_comment_line or is_blank_line:
                definition_line_index_integer += 1
            else:
                break

        # 2c. Find the end of the item via naive brace counting.
        if definition_line_index_integer >= total_file_line_count_integer:
            # Pathological: test attribute with no item after it. Strip
            # at least the attribute line itself.
            region_end_line_index_integer = current_scan_line_index_integer
        else:
            region_end_line_index_integer = (
                _find_matching_close_brace_line_index_via_brace_counting(
                    file_text_lines_list, definition_line_index_integer
                )
            )

        test_regions_to_strip_list.append(
            (region_top_line_index_integer, region_end_line_index_integer)
        )

        # 3. Resume after this stripped region.
        current_scan_line_index_integer = region_end_line_index_integer + 1

    return test_regions_to_strip_list


# ============================================================================
# Output construction
# ============================================================================

def _build_set_of_line_indices_inside_any_region(
    regions_list: list[tuple[int, int]],
) -> set[int]:
    """
    Flatten a list of inclusive (start, end) ranges into a set of all
    contained line indices.

    Parameters
    ----------
    regions_list : list[tuple[int, int]]
        Inclusive (start_line, end_line) ranges.

    Returns
    -------
    set[int]
        Set of every line index in any region.
    """
    accumulated_line_index_set: set[int] = set()
    for (range_start_line_index_integer, range_end_line_index_integer) in regions_list:
        for single_index_in_range_integer in range(
            range_start_line_index_integer, range_end_line_index_integer + 1
        ):
            accumulated_line_index_set.add(single_index_in_range_integer)
    return accumulated_line_index_set


def _build_no_tests_output_lines_from_source_lines(
    file_text_lines_list: list[str],
) -> list[str]:
    """
    Apply ONLY the test-stripping transformation to produce the
    tests-removed version of a Rust source file's contents.

    Parameters
    ----------
    file_text_lines_list : list[str]
        The full content of a `.rs` file split into lines (line endings
        preserved).

    Returns
    -------
    list[str]
        The transformed lines, ready to be written to the output file.
        Line endings are preserved as found in the source.
    """
    # 1. Find test regions to strip entirely.
    test_regions_to_strip_list = _find_test_regions_to_strip_from_file(
        file_text_lines_list
    )
    line_indices_to_skip_entirely_set = _build_set_of_line_indices_inside_any_region(
        test_regions_to_strip_list
    )

    # 2. Walk the original lines and emit everything that is NOT in a
    #    stripped region.
    no_tests_output_lines_list: list[str] = []
    for line_index_integer, single_line_string in enumerate(file_text_lines_list):
        if line_index_integer in line_indices_to_skip_entirely_set:
            continue
        no_tests_output_lines_list.append(single_line_string)

    return no_tests_output_lines_list


def _build_no_tests_output_filename_for_source_file(
    source_file_absolute_path: str,
    source_root_absolute_path_or_none: str | None,
    filename_prefix: str,
) -> str:
    """
    Build the output filename (basename only) for a given Rust source
    file.

    If `source_root_absolute_path_or_none` is provided AND the source
    file is below that root, the relative path from the root is encoded
    into the filename (path separators replaced with `__`) so that
    multiple source files with the same basename do not collide.

    Parameters
    ----------
    source_file_absolute_path : str
        Absolute path to the `.rs` source file.
    source_root_absolute_path_or_none : str | None
        Optional root directory; if given, included in the filename.
    filename_prefix : str
        Prefix to prepend (e.g. `'no_tests.'`).

    Returns
    -------
    str
        Output filename (no directory component), e.g.
        `'no_tests.src__lib__main.rs'` or `'no_tests.main.rs'`.
    """
    if source_root_absolute_path_or_none is not None:
        try:
            relative_path_from_root_string = os.path.relpath(
                source_file_absolute_path, source_root_absolute_path_or_none
            )
        except ValueError:
            # Different drive on Windows etc.; fall back to basename.
            relative_path_from_root_string = os.path.basename(source_file_absolute_path)
    else:
        relative_path_from_root_string = os.path.basename(source_file_absolute_path)

    # Encode path separators into the filename.
    encoded_relative_path_string = relative_path_from_root_string.replace(
        os.sep, "__"
    ).replace("/", "__")
    # Strip any leading "..__" patterns.
    while encoded_relative_path_string.startswith(".."):
        encoded_relative_path_string = encoded_relative_path_string[2:].lstrip("_")

    return f"{filename_prefix}{encoded_relative_path_string}"


def _write_no_tests_lines_to_unique_output_file(
    output_directory_absolute_path: str,
    base_output_filename_string: str,
    no_tests_lines_list: list[str],
    source_file_absolute_path: str,
    error_log_absolute_file_path: str,
) -> str:
    """
    Write a list of output lines to a uniquely-named file in the output
    directory. On filename collision, retry with a numeric suffix.

    Parameters
    ----------
    output_directory_absolute_path : str
        Directory in which to write the file.
    base_output_filename_string : str
        Initial filename to try.
    no_tests_lines_list : list[str]
        Lines to write (newlines should already be present).
    source_file_absolute_path : str
        Original source path (used for the provenance header).
    error_log_absolute_file_path : str
        For logging on write failure.

    Returns
    -------
    str
        Absolute path of the file actually written.

    Raises
    ------
    OSError
        If no unique filename can be obtained after multiple retries or
        if the file write itself fails.
    """
    candidate_full_path_string = os.path.join(
        output_directory_absolute_path, base_output_filename_string
    )

    maximum_collision_retry_attempts_integer = 50
    collision_retry_attempt_counter_integer = 0
    while os.path.exists(candidate_full_path_string):
        collision_retry_attempt_counter_integer += 1
        if collision_retry_attempt_counter_integer > maximum_collision_retry_attempts_integer:
            raise OSError(
                "Could not obtain a unique output filename after "
                f"{maximum_collision_retry_attempts_integer} attempts for "
                f"base name {base_output_filename_string!r} in directory "
                f"{output_directory_absolute_path!r}."
            )
        time.sleep(0.01)
        # Insert a numeric suffix before the final `.rs` extension if
        # possible, otherwise append.
        if base_output_filename_string.endswith(".rs"):
            stem_without_extension_string = base_output_filename_string[:-3]
            candidate_filename_with_suffix_string = (
                f"{stem_without_extension_string}"
                f"__dup{collision_retry_attempt_counter_integer}.rs"
            )
        else:
            candidate_filename_with_suffix_string = (
                f"{base_output_filename_string}"
                f"__dup{collision_retry_attempt_counter_integer}"
            )
        candidate_full_path_string = os.path.join(
            output_directory_absolute_path,
            candidate_filename_with_suffix_string,
        )

    # Provenance header so consumers know what this file is.
    provenance_header_text_string = (
        "// Output file produced by no_tests_rs.\n"
        f"// Source file: {source_file_absolute_path}\n"
        f"// Generated at: {datetime.now().isoformat(timespec='seconds')}\n"
        "// NOTE: cargo test code stripped. Function bodies and all other\n"
        "//       non-test items are preserved verbatim. Original unchanged.\n"
        "\n"
    )

    try:
        with open(
            candidate_full_path_string, "w", encoding="utf-8"
        ) as output_file_handle:
            output_file_handle.write(provenance_header_text_string)
            for single_output_line_string in no_tests_lines_list:
                output_file_handle.write(single_output_line_string)
                if not single_output_line_string.endswith("\n"):
                    output_file_handle.write("\n")
    except Exception as file_write_exception_object:
        full_traceback_text_string = traceback.format_exc()
        _log_error_message_to_file_and_terminal(
            error_log_absolute_file_path,
            (
                "Failed to write no-tests output file "
                f"{candidate_full_path_string!r} (source: "
                f"{source_file_absolute_path!r}): "
                f"{file_write_exception_object!r}\n"
                f"{full_traceback_text_string}"
            ),
        )
        raise

    return candidate_full_path_string


def _process_single_rust_source_file(
    source_file_absolute_path: str,
    source_root_absolute_path_or_none: str | None,
    output_directory_absolute_path: str,
    filename_prefix: str,
    error_log_absolute_file_path: str,
) -> str | None:
    """
    Read one `.rs` file, strip its tests, and write the result to the
    output directory.

    Parameters
    ----------
    source_file_absolute_path : str
        Absolute path to the source `.rs` file (NOT modified).
    source_root_absolute_path_or_none : str | None
        Root directory used when constructing the output filename so
        that nested paths are encoded uniquely.
    output_directory_absolute_path : str
        Directory into which the output file is written.
    filename_prefix : str
        Filename prefix (e.g. `'no_tests.'`).
    error_log_absolute_file_path : str
        Path used to log any errors that occur during processing.

    Returns
    -------
    str | None
        Absolute path of the produced output file, or None on failure
        (failure is logged; no exception is raised).
    """
    try:
        with open(
            source_file_absolute_path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as source_file_handle:
            file_text_lines_list = source_file_handle.readlines()
    except Exception as file_read_exception_object:
        full_traceback_text_string = traceback.format_exc()
        _log_error_message_to_file_and_terminal(
            error_log_absolute_file_path,
            (
                f"Failed to read source file {source_file_absolute_path!r}: "
                f"{file_read_exception_object!r}\n"
                f"{full_traceback_text_string}"
            ),
        )
        return None

    try:
        no_tests_lines_list = _build_no_tests_output_lines_from_source_lines(
            file_text_lines_list
        )
    except Exception as build_exception_object:
        full_traceback_text_string = traceback.format_exc()
        _log_error_message_to_file_and_terminal(
            error_log_absolute_file_path,
            (
                "Failed to build no-tests output for source file "
                f"{source_file_absolute_path!r}: "
                f"{build_exception_object!r}\n"
                f"{full_traceback_text_string}"
            ),
        )
        return None

    base_output_filename_string = _build_no_tests_output_filename_for_source_file(
        source_file_absolute_path=source_file_absolute_path,
        source_root_absolute_path_or_none=source_root_absolute_path_or_none,
        filename_prefix=filename_prefix,
    )

    try:
        written_path_string = _write_no_tests_lines_to_unique_output_file(
            output_directory_absolute_path=output_directory_absolute_path,
            base_output_filename_string=base_output_filename_string,
            no_tests_lines_list=no_tests_lines_list,
            source_file_absolute_path=source_file_absolute_path,
            error_log_absolute_file_path=error_log_absolute_file_path,
        )
    except Exception:
        # _write_no_tests_lines_to_unique_output_file already logged.
        # Swallow here so other files in the batch can still be processed.
        return None

    return written_path_string


# ============================================================================
# Public API
# ============================================================================

def no_tests_rs(
    rust_code_dir_path: str | None = None,
    only_process_this_file_path: str | None = None,
    file_depth: int = 2,
    output_dir: str | None = None,
    output_filename_prefix: str = DEFAULT_NO_TESTS_FILENAME_PREFIX,
) -> list[str]:
    """
    Produce tests-removed copies of one or more Rust source files. The
    original source files are NEVER modified.

    Parameters
    ----------
    rust_code_dir_path : str | None, default None
        Path to a directory containing `.rs` files to process.
        Required unless `only_process_this_file_path` is provided.
    only_process_this_file_path : str | None, default None
        If provided, process only this single `.rs` file (ignores
        `rust_code_dir_path` and `file_depth` for discovery).
    file_depth : int, default 2
        Maximum nested-directory depth to descend below
        `rust_code_dir_path` when discovering `.rs` files. 0 = only
        files directly in the root.
    output_dir : str | None, default None
        Directory in which the `no_tests_files_<timestamp>/`
        subdirectory will be created. If None, the current working
        directory is used.
    output_filename_prefix : str, default 'no_tests.'
        Prefix prepended to each output filename.

    Returns
    -------
    list[str]
        Absolute paths of ALL files produced by this run: every
        tests-removed output file plus the run's error log file (the
        error log is included even if empty).

    Output files
    ------------
    Files are written under:
        {output_dir or cwd}/no_tests_files_{ISO8601_timestamp}/

    With names:
        {output_filename_prefix}{relative_path_with___sep}.rs
        no_tests_error_log_{ISO8601_timestamp}.txt

    Raises
    ------
    ValueError
        If neither `rust_code_dir_path` nor `only_process_this_file_path`
        is provided, or if argument types/values are invalid.
    FileNotFoundError
        If the provided directory or file does not exist.
    OSError
        If the output directory cannot be created.

    Behavior on errors
    ------------------
    Per-file errors are logged (to the error log file and the terminal)
    and the run continues with the remaining files. The error log file
    path is always included in the returned list.

    Example
    -------
    >>> produced_paths_list = no_tests_rs(
    ...     rust_code_dir_path="./my_rust_project/src",
    ...     file_depth=3,
    ... )
    >>> for single_produced_path in produced_paths_list:
    ...     print(single_produced_path)
    """
    # --- 0. Validate inputs early with clear messages ---------------------
    if rust_code_dir_path is None and only_process_this_file_path is None:
        raise ValueError(
            "no_tests_rs requires either `rust_code_dir_path` or "
            "`only_process_this_file_path` to be provided (both were None)."
        )

    if rust_code_dir_path is not None:
        if not isinstance(rust_code_dir_path, str) or not rust_code_dir_path:
            raise ValueError(
                "no_tests_rs: `rust_code_dir_path` must be a non-empty "
                f"string when provided; got {rust_code_dir_path!r}."
            )
        if not os.path.isdir(rust_code_dir_path):
            raise FileNotFoundError(
                "no_tests_rs: `rust_code_dir_path` does not exist or is "
                f"not a directory: {rust_code_dir_path!r}"
            )

    if only_process_this_file_path is not None:
        if (
            not isinstance(only_process_this_file_path, str)
            or not only_process_this_file_path
        ):
            raise ValueError(
                "no_tests_rs: `only_process_this_file_path` must be a "
                f"non-empty string when provided; got "
                f"{only_process_this_file_path!r}."
            )
        if not os.path.isfile(only_process_this_file_path):
            raise FileNotFoundError(
                "no_tests_rs: `only_process_this_file_path` does not exist "
                f"or is not a file: {only_process_this_file_path!r}"
            )

    if not isinstance(file_depth, int) or file_depth < 0:
        raise ValueError(
            "no_tests_rs: `file_depth` must be a non-negative integer; "
            f"got {file_depth!r}."
        )

    if not isinstance(output_filename_prefix, str):
        raise ValueError(
            "no_tests_rs: `output_filename_prefix` must be a string; "
            f"got {output_filename_prefix!r}."
        )

    # --- 1. Resolve output directory and create timestamped subdir -------
    effective_output_root_directory_string = (
        output_dir if output_dir is not None else os.getcwd()
    )
    if not os.path.isdir(effective_output_root_directory_string):
        raise FileNotFoundError(
            "no_tests_rs: `output_dir` does not exist or is not a "
            f"directory: {effective_output_root_directory_string!r}"
        )

    # Timestamped subdirectory name, e.g. 'no_tests_files_2024-01-15T14_32_45_123456'.
    run_timestamp_string = _generate_iso8601_timestamp_string()
    no_tests_subdirectory_basename_string = (
        f"{DEFAULT_OUTPUT_SUBDIRECTORY_NAME}_{run_timestamp_string}"
    )
    no_tests_output_subdirectory_absolute_path = os.path.abspath(
        os.path.join(
            effective_output_root_directory_string,
            no_tests_subdirectory_basename_string,
        )
    )
    try:
        os.makedirs(
            no_tests_output_subdirectory_absolute_path, exist_ok=True
        )
    except Exception as output_dir_creation_exception_object:
        full_traceback_text_string = traceback.format_exc()
        raise OSError(
            "no_tests_rs: failed to create output directory "
            f"{no_tests_output_subdirectory_absolute_path!r}: "
            f"{output_dir_creation_exception_object!r}\n"
            f"{full_traceback_text_string}"
        ) from output_dir_creation_exception_object

    # --- 2. Establish run-wide error log ---------------------------------
    error_log_absolute_file_path = os.path.join(
        no_tests_output_subdirectory_absolute_path,
        f"no_tests_error_log_{run_timestamp_string}.txt",
    )
    try:
        with open(
            error_log_absolute_file_path, "w", encoding="utf-8"
        ) as initial_log_file_handle:
            initial_log_file_handle.write(
                "no_tests_rs error log\n"
                f"Timestamp: {run_timestamp_string}\n"
                f"rust_code_dir_path: {rust_code_dir_path!r}\n"
                f"only_process_this_file_path: "
                f"{only_process_this_file_path!r}\n"
                f"file_depth: {file_depth}\n"
                f"output_dir: {output_dir!r}\n"
                f"output_filename_prefix: {output_filename_prefix!r}\n"
                f"{'-' * 80}\n"
            )
    except Exception as log_init_exception_object:
        # Non-fatal; we will keep trying via the logger helper.
        print(
            "[no_tests_rs ERROR] Could not initialize error log file at "
            f"{error_log_absolute_file_path!r}: {log_init_exception_object!r}"
        )

    # --- 3. Determine the list of source files to process ----------------
    source_root_absolute_path_or_none: str | None
    files_to_process_absolute_paths_list: list[str] = []
    try:
        if only_process_this_file_path is not None:
            single_absolute_path_string = os.path.abspath(
                only_process_this_file_path
            )
            files_to_process_absolute_paths_list = [single_absolute_path_string]
            # Use the file's parent directory as the "root" for filename
            # encoding, so the output is `no_tests.<basename>.rs`.
            source_root_absolute_path_or_none = os.path.dirname(
                single_absolute_path_string
            )
        else:
            # rust_code_dir_path guaranteed non-None at this point.
            assert rust_code_dir_path is not None  # for type checkers
            source_root_absolute_path_or_none = os.path.abspath(
                rust_code_dir_path
            )
            files_to_process_absolute_paths_list = (
                _collect_rust_source_files_with_depth_limit(
                    source_root_absolute_path_or_none, file_depth
                )
            )
    except Exception as discovery_exception_object:
        full_traceback_text_string = traceback.format_exc()
        _log_error_message_to_file_and_terminal(
            error_log_absolute_file_path,
            (
                "Failed during source-file discovery: "
                f"{discovery_exception_object!r}\n"
                f"{full_traceback_text_string}"
            ),
        )
        # Return what we have (just the log file).
        return [error_log_absolute_file_path]

    if not files_to_process_absolute_paths_list:
        _log_error_message_to_file_and_terminal(
            error_log_absolute_file_path,
            (
                "No `.rs` files found to process. "
                f"rust_code_dir_path={rust_code_dir_path!r}, "
                f"only_process_this_file_path="
                f"{only_process_this_file_path!r}, "
                f"file_depth={file_depth}."
            ),
        )

    # --- 4. Process each file --------------------------------------------
    produced_output_file_paths_list: list[str] = []
    for single_source_file_absolute_path_string in files_to_process_absolute_paths_list:
        try:
            written_path_or_none = _process_single_rust_source_file(
                source_file_absolute_path=single_source_file_absolute_path_string,
                source_root_absolute_path_or_none=source_root_absolute_path_or_none,
                output_directory_absolute_path=no_tests_output_subdirectory_absolute_path,
                filename_prefix=output_filename_prefix,
                error_log_absolute_file_path=error_log_absolute_file_path,
            )
            if written_path_or_none is not None:
                produced_output_file_paths_list.append(written_path_or_none)
        except Exception as per_file_processing_exception_object:
            # Defensive: the helper already catches and logs, but in case
            # something escapes, log here too and keep going.
            full_traceback_text_string = traceback.format_exc()
            _log_error_message_to_file_and_terminal(
                error_log_absolute_file_path,
                (
                    "Unexpected exception while processing source file "
                    f"{single_source_file_absolute_path_string!r}: "
                    f"{per_file_processing_exception_object!r}\n"
                    f"{full_traceback_text_string}"
                ),
            )

    # --- 5. Always include the error log in the returned list ------------
    produced_output_file_paths_list.append(error_log_absolute_file_path)
    return produced_output_file_paths_list


# ============================================================================
# CLI Q&A
# ============================================================================

if __name__ == "__main__":
    # Simple interactive Q&A, mirroring function_finder.py /
    # functional_table_of_contents.py.
    print(
        "no_tests_rs module. Import and call:\n"
        "    from no_tests_rs import no_tests_rs\n"
        "    produced_paths_list = no_tests_rs(\n"
        "        rust_code_dir_path='./path/to/rust/src',\n"
        "        only_process_this_file_path=None,\n"
        "        file_depth=2,\n"
        "        output_dir=None,\n"
        "        output_filename_prefix='no_tests.',\n"
        "    )\n"
    )

    print("\nNo Tests (Rust) Q&A\n")

    rust_code_dir_path_input_string = input(
        "What is rust_code_dir_path? (leave blank if using a single file)\n > "
    )
    only_process_this_file_path_input_string = input(
        "What is only_process_this_file_path? (leave blank if using a directory)\n > "
    )
    file_depth_input_string = input(
        "(optional) What is file_depth? (default 2)\n > "
    )
    output_dir_input_string = input(
        "(optional) What is output_dir? (default cwd)\n > "
    )
    output_filename_prefix_input_string = input(
        "(optional) What is output_filename_prefix? (default 'no_tests.')\n > "
    )

    # Normalize blanks to None / defaults.
    if rust_code_dir_path_input_string.strip() == "":
        rust_code_dir_path_resolved = None
    else:
        rust_code_dir_path_resolved = rust_code_dir_path_input_string.strip()

    if only_process_this_file_path_input_string.strip() == "":
        only_process_this_file_path_resolved = None
    else:
        only_process_this_file_path_resolved = (
            only_process_this_file_path_input_string.strip()
        )

    if file_depth_input_string.strip() == "":
        file_depth_resolved = 2
    else:
        try:
            file_depth_resolved = int(file_depth_input_string.strip())
        except ValueError:
            print(
                f"Could not parse file_depth={file_depth_input_string!r} as "
                "an int; using default of 2."
            )
            file_depth_resolved = 2

    if output_dir_input_string.strip() == "":
        output_dir_resolved = None
    else:
        output_dir_resolved = output_dir_input_string.strip()

    if output_filename_prefix_input_string.strip() == "":
        output_filename_prefix_resolved = DEFAULT_NO_TESTS_FILENAME_PREFIX
    else:
        output_filename_prefix_resolved = output_filename_prefix_input_string

    try:
        produced_paths_list = no_tests_rs(
            rust_code_dir_path=rust_code_dir_path_resolved,
            only_process_this_file_path=only_process_this_file_path_resolved,
            file_depth=file_depth_resolved,
            output_dir=output_dir_resolved,
            output_filename_prefix=output_filename_prefix_resolved,
        )
    except Exception as top_level_cli_exception_object:
        full_traceback_text_string = traceback.format_exc()
        print(
            "[no_tests_rs CLI ERROR] "
            f"{top_level_cli_exception_object!r}\n{full_traceback_text_string}"
        )
        produced_paths_list = []

    print("\nProduced files:")
    for single_produced_path_string in produced_paths_list:
        print(f"  {single_produced_path_string}")

    print("\nNo Tests Ok!\n")
