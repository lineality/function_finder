"""
functional_table_of_contents module
===================================

Produces "table of contents" SUMMARY files for Rust source code.

For every input `.rs` file, this module writes a parallel summary file
that:
  1. Contains the original file's structure (uses, mods, structs, enums,
     traits, impls, type aliases, constants, statics, doc-comments,
     attributes).
  2. Has all cargo-test code stripped out (e.g. `#[cfg(test)] mod tests
     { ... }` and standalone `#[test] fn ...`).
  3. Has every remaining function's BODY replaced with a short
     placeholder line, while the function's signature, doc-comments, and
     attributes are preserved verbatim.

The original source file is NEVER modified. Each summary is written to a
separate output file (default: `functional_toc_files/toc.{name}.rs`).

Public API
----------
    functional_table_of_contents(...)
        Process a single `.rs` file or a directory of `.rs` files.

Design notes / known limitations
--------------------------------
- This is a line-based heuristic tool, NOT a Rust AST parser. It mirrors
  the design of `function_finder.py`.
- Brace counting is naive. Braces inside string literals, character
  literals, raw strings, comments, and `macro_rules!` bodies may confuse
  the counter. Accepted per spec: "the goal is not to be 1000%
  impossibly perfect."
- Multi-line attributes (`#[derive(\n    Foo,\n)]`) are only partially
  handled: each line starting with `#[` is treated as an attribute line.
- Doc comments are detected via `///` and `//!` line prefixes. Block
  doc comments (`/** */`) are not specifically handled.

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

# The single-line placeholder that replaces a function's body.
# Surrounding braces are included so that, regardless of how messy the
# original signature line is, the produced summary always contains a
# syntactically-plausible balanced `{ ... }` pair for the function.
DEFAULT_FUNCTION_BODY_PLACEHOLDER_LINE: str = (
    "    /* ... body removed by functional_table_of_contents ... */\n"
)

# Default filename prefix for the produced summary files.
# Example: source "foo.rs" -> summary "toc.foo.rs".
DEFAULT_SUMMARY_FILENAME_PREFIX: str = "toc."



# Default subdirectory (created inside cwd or `output_dir`) into which
# summary files are written.
DEFAULT_OUTPUT_SUBDIRECTORY_NAME: str = "functional_toc_files"

# Attribute substrings that mark a Rust item as test-only. Any line
# whose stripped form CONTAINS one of these substrings is treated as
# the start (or part) of a test region to be stripped from the summary.
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
    print(f"[functional_table_of_contents ERROR] {error_message_text}")
    try:
        with open(
            error_log_absolute_file_path, "a", encoding="utf-8"
        ) as error_log_file_handle:
            error_log_file_handle.write(error_message_text)
            error_log_file_handle.write("\n" + ("-" * 80) + "\n")
    except Exception as inner_log_write_exception:
        # Swallow secondary failures; we already printed the primary error.
        print(
            "[functional_table_of_contents ERROR] Additionally, failed to "
            f"write the above error to the log file at "
            f"{error_log_absolute_file_path!r}: "
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
# Rust-specific detection regexes
# ============================================================================

def _build_any_function_definition_detection_regex() -> "re.Pattern[str]":
    """
    Build a compiled regex that matches a line which begins (after
    optional leading whitespace) with a Rust function definition of ANY
    name.

    Returns
    -------
    re.Pattern[str]
        Compiled regex. A match indicates the line starts a function
        definition (`fn <identifier>` with optional visibility/modifier
        prefixes).

    Notes
    -----
    Accepts the same visibility/modifier prefixes as
    `function_finder.py`:
      - visibility: `pub`, `pub(crate)`, `pub(super)`, `pub(in path)`
      - modifiers (any combination/order): `async`, `unsafe`, `const`,
        `extern "ABI"`, `default`
    """
    # Visibility (optional).
    visibility_pattern_fragment = r"(?:pub(?:\([^)]*\))?\s+)?"
    # Modifier keywords (each optional, any order).
    modifier_pattern_fragment = (
        r"(?:(?:async|unsafe|const|default)\s+)*"
        r"(?:extern\s+(?:\"[^\"]*\"\s+)?)?"
        r"(?:(?:async|unsafe|const|default)\s+)*"
    )
    # The function name itself (any valid Rust identifier).
    function_name_pattern_fragment = r"[A-Za-z_][A-Za-z0-9_]*"

    full_regex_pattern_string = (
        r"^\s*"
        + visibility_pattern_fragment
        + modifier_pattern_fragment
        + r"fn\s+"
        + function_name_pattern_fragment
        + r"\s*[<(]"
    )
    return re.compile(full_regex_pattern_string)


# Pre-compile once at module load for reuse.
_COMPILED_ANY_FUNCTION_DEFINITION_REGEX = _build_any_function_definition_detection_regex()


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
        the line index of that `;` is returned (caller can interpret
        this as "no body"). If no balanced close brace is found, the
        last line index of the file is returned as a safe fallback.

    Limitations
    -----------
    Braces inside string literals, character literals, raw strings, or
    comments will mislead this counter. This is accepted per the
    project spec.
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

        # Signature-only line (e.g. trait method declaration).
        if (
            not have_seen_first_opening_brace_flag
            and single_line_string.rstrip().endswith(";")
        ):
            return current_line_index_integer

        current_line_index_integer += 1

    return total_file_line_count_integer - 1


def _find_function_body_open_and_close_brace_line_indices(
    file_text_lines_list: list[str],
    function_definition_line_index: int,
) -> tuple[int, int] | None:
    """
    Locate the line indices of the opening `{` and matching closing `}`
    of a function's body, starting from the function's definition line.

    Parameters
    ----------
    file_text_lines_list : list[str]
        The full content of a `.rs` file split into lines.
    function_definition_line_index : int
        Zero-indexed line where the function's `fn name(...)` begins.

    Returns
    -------
    tuple[int, int] | None
        (open_brace_line_index, close_brace_line_index) if the function
        has a body. None if the signature is bodyless (ends in `;`, e.g.
        a trait method declaration) or if no balanced body can be
        located.
    """
    running_brace_balance_integer = 0
    have_seen_first_opening_brace_flag = False
    open_brace_line_index_or_none: int | None = None
    total_file_line_count_integer = len(file_text_lines_list)

    current_line_index_integer = function_definition_line_index
    while current_line_index_integer < total_file_line_count_integer:
        single_line_string = file_text_lines_list[current_line_index_integer]

        for single_character_in_line in single_line_string:
            if single_character_in_line == "{":
                if not have_seen_first_opening_brace_flag:
                    open_brace_line_index_or_none = current_line_index_integer
                    have_seen_first_opening_brace_flag = True
                running_brace_balance_integer += 1
            elif single_character_in_line == "}":
                running_brace_balance_integer -= 1
                if (
                    have_seen_first_opening_brace_flag
                    and running_brace_balance_integer == 0
                ):
                    # Found matching close.
                    if open_brace_line_index_or_none is None:
                        return None  # Defensive; should never happen.
                    return (
                        open_brace_line_index_or_none,
                        current_line_index_integer,
                    )

        # Bodyless signature (e.g. `fn foo(&self) -> u32;` inside a trait).
        if (
            not have_seen_first_opening_brace_flag
            and single_line_string.rstrip().endswith(";")
        ):
            return None

        current_line_index_integer += 1

    return None


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
        True if the line contains any of the substrings in
        `TEST_MARKER_ATTRIBUTE_SUBSTRINGS_TUPLE` AND begins with `#[`.
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
    should be removed entirely from the summary.

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
# Function body shrink planning
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


def _find_function_body_shrink_plans_for_file(
    file_text_lines_list: list[str],
    line_indices_to_skip_entirely_set: set[int],
) -> list[tuple[int, int, int]]:
    """
    Identify every function definition (outside of skip regions) whose
    body should be replaced with the placeholder line.

    Parameters
    ----------
    file_text_lines_list : list[str]
        The full content of a `.rs` file split into lines.
    line_indices_to_skip_entirely_set : set[int]
        Line indices that are already going to be stripped (e.g.
        because they belong to a test region). Functions whose
        definition line is in this set are ignored to avoid
        double-processing.

    Returns
    -------
    list[tuple[int, int, int]]
        Each tuple is (open_brace_line_index, close_brace_line_index,
        insert_placeholder_after_line_index). The orchestrator will
        skip lines strictly BETWEEN open and close, and insert the
        placeholder line immediately after `insert_placeholder_after_line_index`.

        Trivial cases (body entirely on the open-brace line, or no
        interior lines to remove) are filtered out: no shrink plan is
        emitted for them.
    """
    function_body_shrink_plans_list: list[tuple[int, int, int]] = []

    for line_index_integer, single_line_string in enumerate(file_text_lines_list):
        if line_index_integer in line_indices_to_skip_entirely_set:
            continue
        if not _COMPILED_ANY_FUNCTION_DEFINITION_REGEX.match(single_line_string):
            continue

        # Locate the body's open and close braces.
        brace_span_or_none = _find_function_body_open_and_close_brace_line_indices(
            file_text_lines_list, line_index_integer
        )
        if brace_span_or_none is None:
            # Bodyless signature (trait method) -- leave untouched.
            continue

        open_brace_line_index_integer, close_brace_line_index_integer = brace_span_or_none

        # If body is on a single line, or there are no interior lines to
        # remove, leave the function alone.
        if close_brace_line_index_integer <= open_brace_line_index_integer + 1:
            continue

        function_body_shrink_plans_list.append(
            (
                open_brace_line_index_integer,
                close_brace_line_index_integer,
                open_brace_line_index_integer,
            )
        )

    return function_body_shrink_plans_list


# ============================================================================
# Summary file construction
# ============================================================================

def _build_summary_text_lines_from_source_lines(
    file_text_lines_list: list[str],
) -> list[str]:
    """
    Apply the test-stripping and function-body-shrinking transformations
    to produce the summary version of a Rust source file's contents.

    Parameters
    ----------
    file_text_lines_list : list[str]
        The full content of a `.rs` file split into lines (with line
        endings preserved).

    Returns
    -------
    list[str]
        The transformed lines, ready to be written to the summary file.
        Lines are returned with line endings preserved where possible;
        the placeholder line carries its own trailing newline.
    """
    # 1. Find test regions to strip entirely.
    test_regions_to_strip_list = _find_test_regions_to_strip_from_file(
        file_text_lines_list
    )
    line_indices_to_skip_entirely_set = _build_set_of_line_indices_inside_any_region(
        test_regions_to_strip_list
    )

    # 2. Find function bodies to collapse (only those NOT in test regions).
    function_body_shrink_plans_list = _find_function_body_shrink_plans_for_file(
        file_text_lines_list, line_indices_to_skip_entirely_set
    )

    # 3. Build the additional skip set from function-body interiors, and
    #    build the "insert placeholder after this line" map.
    placeholder_insertion_after_line_map: dict[int, str] = {}
    for (
        open_brace_line_index_integer,
        close_brace_line_index_integer,
        insert_after_line_index_integer,
    ) in function_body_shrink_plans_list:
        # Skip the interior of the body: open+1 .. close-1 inclusive.
        for interior_line_index_integer in range(
            open_brace_line_index_integer + 1, close_brace_line_index_integer
        ):
            line_indices_to_skip_entirely_set.add(interior_line_index_integer)
        # Insert placeholder line after the open brace line.
        placeholder_insertion_after_line_map[insert_after_line_index_integer] = (
            DEFAULT_FUNCTION_BODY_PLACEHOLDER_LINE
        )

    # 4. Walk the original lines and emit the summary.
    summary_output_lines_list: list[str] = []
    for line_index_integer, single_line_string in enumerate(file_text_lines_list):
        if line_index_integer in line_indices_to_skip_entirely_set:
            continue
        summary_output_lines_list.append(single_line_string)
        if line_index_integer in placeholder_insertion_after_line_map:
            summary_output_lines_list.append(
                placeholder_insertion_after_line_map[line_index_integer]
            )

    return summary_output_lines_list


def _build_summary_output_filename_for_source_file(
    source_file_absolute_path: str,
    source_root_absolute_path_or_none: str | None,
    filename_prefix: str,
) -> str:
    """
    Build the summary output filename (basename only) for a given Rust
    source file.

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
        Prefix to prepend (e.g. `'toc.'`).

    Returns
    -------
    str
        Output filename (no directory component), e.g.
        `'toc.src__lib__main.rs'` or `'toc.main.rs'`.
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
    # Also strip any leading "..__" patterns just to be safe.
    while encoded_relative_path_string.startswith(".."):
        encoded_relative_path_string = encoded_relative_path_string[2:].lstrip("_")

    return f"{filename_prefix}{encoded_relative_path_string}"


def _write_summary_lines_to_unique_output_file(
    output_directory_absolute_path: str,
    base_output_filename_string: str,
    summary_lines_list: list[str],
    source_file_absolute_path: str,
    error_log_absolute_file_path: str,
) -> str:
    """
    Write a list of summary lines to a uniquely-named output file. On
    filename collision, retry with a numeric suffix.

    Parameters
    ----------
    output_directory_absolute_path : str
        Directory in which to write the file.
    base_output_filename_string : str
        Initial filename to try.
    summary_lines_list : list[str]
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
        If no unique filename could be obtained after multiple retries
        or if the file write itself fails.
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
                "Could not obtain a unique summary filename after "
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

    # Build provenance header.
    provenance_header_text_string = (
        "// Summary file produced by functional_table_of_contents.\n"
        f"// Source file: {source_file_absolute_path}\n"
        f"// Generated at: {datetime.now().isoformat(timespec='seconds')}\n"
        "// NOTE: cargo test code stripped; function bodies replaced with\n"
        "//       a placeholder. The original file is unchanged.\n"
        "\n"
    )

    try:
        with open(
            candidate_full_path_string, "w", encoding="utf-8"
        ) as output_file_handle:
            output_file_handle.write(provenance_header_text_string)
            for single_output_line_string in summary_lines_list:
                output_file_handle.write(single_output_line_string)
                if not single_output_line_string.endswith("\n"):
                    output_file_handle.write("\n")
    except Exception as file_write_exception_object:
        full_traceback_text_string = traceback.format_exc()
        _log_error_message_to_file_and_terminal(
            error_log_absolute_file_path,
            (
                "Failed to write summary file "
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
    Read one `.rs` file, build its summary, and write the summary to
    the output directory.

    Parameters
    ----------
    source_file_absolute_path : str
        Absolute path to the source `.rs` file (NOT modified).
    source_root_absolute_path_or_none : str | None
        Root directory used when constructing the output filename so
        that nested paths are encoded uniquely.
    output_directory_absolute_path : str
        Directory into which the summary file is written.
    filename_prefix : str
        Filename prefix (e.g. `'toc.'`).
    error_log_absolute_file_path : str
        Path used to log any errors that occur during processing.

    Returns
    -------
    str | None
        Absolute path of the produced summary file, or None on failure
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
        summary_lines_list = _build_summary_text_lines_from_source_lines(
            file_text_lines_list
        )
    except Exception as summary_build_exception_object:
        full_traceback_text_string = traceback.format_exc()
        _log_error_message_to_file_and_terminal(
            error_log_absolute_file_path,
            (
                "Failed to build summary for source file "
                f"{source_file_absolute_path!r}: "
                f"{summary_build_exception_object!r}\n"
                f"{full_traceback_text_string}"
            ),
        )
        return None

    base_output_filename_string = _build_summary_output_filename_for_source_file(
        source_file_absolute_path=source_file_absolute_path,
        source_root_absolute_path_or_none=source_root_absolute_path_or_none,
        filename_prefix=filename_prefix,
    )

    try:
        written_path_string = _write_summary_lines_to_unique_output_file(
            output_directory_absolute_path=output_directory_absolute_path,
            base_output_filename_string=base_output_filename_string,
            summary_lines_list=summary_lines_list,
            source_file_absolute_path=source_file_absolute_path,
            error_log_absolute_file_path=error_log_absolute_file_path,
        )
    except Exception as write_exception_object:
        # _write_summary_lines_to_unique_output_file already logged.
        # Swallow here so other files in the batch can still be processed.
        return None

    return written_path_string


# ============================================================================
# Public API
# ============================================================================

def functional_table_of_contents(
    rust_code_dir_path: str | None = None,
    only_process_this_file_path: str | None = None,
    file_depth: int = 2,
    output_dir: str | None = None,
    summary_filename_prefix: str = DEFAULT_SUMMARY_FILENAME_PREFIX,
) -> list[str]:
    """
    Produce table-of-contents summary files for one or more Rust source
    files. The original source files are NEVER modified.

    Parameters
    ----------
    rust_code_dir_path : str | None, default None
        Path to a directory containing `.rs` files to summarize.
        Required unless `only_process_this_file_path` is provided.
    only_process_this_file_path : str | None, default None
        If provided, process only this single `.rs` file (ignores
        `rust_code_dir_path` and `file_depth` for discovery).
    file_depth : int, default 2
        Maximum nested-directory depth to descend below
        `rust_code_dir_path` when discovering `.rs` files. 0 = only
        files directly in the root.
    output_dir : str | None, default None
        Directory in which the `functional_toc_files/` subdirectory
        will be created. If None, the current working directory is used.
    summary_filename_prefix : str, default 'toc.'
        Prefix prepended to each summary filename.
        Returns
        -------
        list[str]
            Absolute paths of ALL files produced by this run: every
            summary file plus the run's error log file (the error log is
            included even if empty).

        Output files
        ------------
        Files are written under:
            {output_dir or cwd}/functional_toc_files/

        With names:
            {summary_filename_prefix}{relative_path_with___sep}.rs
            toc_error_log_{ISO8601_timestamp}.txt

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
        >>> produced_paths_list = functional_table_of_contents(
        ...     rust_code_dir_path="./my_rust_project/src",
        ...     file_depth=3,
        ... )
        >>> for single_produced_path in produced_paths_list:
        ...     print(single_produced_path)
    """
    # --- 0. Validate inputs early with clear messages ---------------------
    if rust_code_dir_path is None and only_process_this_file_path is None:
        raise ValueError(
            "functional_table_of_contents requires either "
            "`rust_code_dir_path` or `only_process_this_file_path` to be "
            "provided (both were None)."
        )

    if rust_code_dir_path is not None:
        if not isinstance(rust_code_dir_path, str) or not rust_code_dir_path:
            raise ValueError(
                "functional_table_of_contents: `rust_code_dir_path` must be "
                f"a non-empty string when provided; got {rust_code_dir_path!r}."
            )
        if not os.path.isdir(rust_code_dir_path):
            raise FileNotFoundError(
                "functional_table_of_contents: `rust_code_dir_path` does "
                f"not exist or is not a directory: {rust_code_dir_path!r}"
            )

    if only_process_this_file_path is not None:
        if (
            not isinstance(only_process_this_file_path, str)
            or not only_process_this_file_path
        ):
            raise ValueError(
                "functional_table_of_contents: `only_process_this_file_path` "
                f"must be a non-empty string when provided; got "
                f"{only_process_this_file_path!r}."
            )
        if not os.path.isfile(only_process_this_file_path):
            raise FileNotFoundError(
                "functional_table_of_contents: `only_process_this_file_path` "
                f"does not exist or is not a file: "
                f"{only_process_this_file_path!r}"
            )

    if not isinstance(file_depth, int) or file_depth < 0:
        raise ValueError(
            "functional_table_of_contents: `file_depth` must be a "
            f"non-negative integer; got {file_depth!r}."
        )

    if not isinstance(summary_filename_prefix, str):
        raise ValueError(
            "functional_table_of_contents: `summary_filename_prefix` must "
            f"be a string; got {summary_filename_prefix!r}."
        )

    # --- 1. Resolve output directory and create functional_toc_files/ ----
    effective_output_root_directory_string = (
        output_dir if output_dir is not None else os.getcwd()
    )
    if not os.path.isdir(effective_output_root_directory_string):
        raise FileNotFoundError(
            "functional_table_of_contents: `output_dir` does not exist or "
            f"is not a directory: {effective_output_root_directory_string!r}"
        )

    toc_timestamp_string = f"{DEFAULT_OUTPUT_SUBDIRECTORY_NAME}_{_generate_iso8601_timestamp_string()}"

    functional_toc_output_subdirectory_absolute_path = os.path.abspath(
        os.path.join(
            effective_output_root_directory_string,
            toc_timestamp_string,
        )
    )
    try:
        os.makedirs(
            functional_toc_output_subdirectory_absolute_path, exist_ok=True
        )
    except Exception as output_dir_creation_exception_object:
        full_traceback_text_string = traceback.format_exc()
        raise OSError(
            "functional_table_of_contents: failed to create output "
            f"directory {functional_toc_output_subdirectory_absolute_path!r}:"
            f" {output_dir_creation_exception_object!r}\n"
            f"{full_traceback_text_string}"
        ) from output_dir_creation_exception_object

    # --- 2. Establish run-wide shared state ------------------------------
    run_timestamp_string = _generate_iso8601_timestamp_string()
    error_log_absolute_file_path = os.path.join(
        functional_toc_output_subdirectory_absolute_path,
        f"toc_error_log_{run_timestamp_string}.txt",
    )
    try:
        with open(
            error_log_absolute_file_path, "w", encoding="utf-8"
        ) as initial_log_file_handle:
            initial_log_file_handle.write(
                "functional_table_of_contents error log\n"
                f"Timestamp: {run_timestamp_string}\n"
                f"rust_code_dir_path: {rust_code_dir_path!r}\n"
                f"only_process_this_file_path: "
                f"{only_process_this_file_path!r}\n"
                f"file_depth: {file_depth}\n"
                f"output_dir: {output_dir!r}\n"
                f"summary_filename_prefix: {summary_filename_prefix!r}\n"
                f"{'-' * 80}\n"
            )
    except Exception as log_init_exception_object:
        # Non-fatal; we will keep trying via the logger helper.
        print(
            "[functional_table_of_contents ERROR] Could not initialize "
            f"error log file at {error_log_absolute_file_path!r}: "
            f"{log_init_exception_object!r}"
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
            # encoding, so the output is `toc.<basename>.rs`.
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
    produced_summary_file_paths_list: list[str] = []
    for single_source_file_absolute_path_string in files_to_process_absolute_paths_list:
        try:
            written_summary_path_or_none = _process_single_rust_source_file(
                source_file_absolute_path=single_source_file_absolute_path_string,
                source_root_absolute_path_or_none=source_root_absolute_path_or_none,
                output_directory_absolute_path=functional_toc_output_subdirectory_absolute_path,
                filename_prefix=summary_filename_prefix,
                error_log_absolute_file_path=error_log_absolute_file_path,
            )
            if written_summary_path_or_none is not None:
                produced_summary_file_paths_list.append(
                    written_summary_path_or_none
                )
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
    produced_summary_file_paths_list.append(error_log_absolute_file_path)
    return produced_summary_file_paths_list


# ============================================================================
# CLI Q&A
# ============================================================================

if __name__ == "__main__":
    # Simple interactive Q&A for command-line use, mirroring the style
    # of function_finder.py.
    print(
        "functional_table_of_contents module. Import and call:\n"
        "    from functional_table_of_contents import functional_table_of_contents\n"
        "    produced_paths_list = functional_table_of_contents(\n"
        "        rust_code_dir_path='./path/to/rust/src',\n"
        "        only_process_this_file_path=None,\n"
        "        file_depth=2,\n"
        "        output_dir=None,\n"
        "        summary_filename_prefix='toc.',\n"
        "    )\n"
    )

    print("\nFunctional Table of Contents (Rust) Q&A\n")

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
    summary_filename_prefix_input_string = input(
        "(optional) What is summary_filename_prefix? (default 'toc.')\n > "
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

    if summary_filename_prefix_input_string.strip() == "":
        summary_filename_prefix_resolved = DEFAULT_SUMMARY_FILENAME_PREFIX
    else:
        summary_filename_prefix_resolved = summary_filename_prefix_input_string

    try:
        produced_paths_list = functional_table_of_contents(
            rust_code_dir_path=rust_code_dir_path_resolved,
            only_process_this_file_path=only_process_this_file_path_resolved,
            file_depth=file_depth_resolved,
            output_dir=output_dir_resolved,
            summary_filename_prefix=summary_filename_prefix_resolved,
        )
    except Exception as top_level_cli_exception_object:
        full_traceback_text_string = traceback.format_exc()
        print(
            "[functional_table_of_contents CLI ERROR] "
            f"{top_level_cli_exception_object!r}\n{full_traceback_text_string}"
        )
        produced_paths_list = []

    print("\nProduced files:")
    for single_produced_path_string in produced_paths_list:
        print(f"  {single_produced_path_string}")

    print("\nTOC Ok!\n")
