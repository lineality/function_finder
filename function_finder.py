"""
function_finder module
======================

Provides two public functions for extracting Rust function definitions from
`.rs` source files using heuristic line-based parsing (NOT a Rust AST
parser):

    1. function_finder(...)
       Locate one or more Rust functions by name in a directory tree of `.rs`
       files, extract each occurrence (including doc-comments and attribute
       lines immediately preceding the definition, plus the full function
       body), and save each extracted function to its own `.rs` file in an
       output directory `function_finder_files/`. Optionally recurse into
       functions called by the target function up to a configurable depth.

    2. flatten_finder(dir_path)
       Concatenate your results (all `.rs` files from a directory (typically the
       `function_finder_files/` directory produced by `function_finder`)
       into a single flat `.rs` file written to the current working directory.

Design notes / known limitations
--------------------------------
- Brace counting is naive: braces appearing inside string literals or
  comments can confuse the end-of-function detection. This is acceptable
  per spec ("simple brace counting to start with").
- Multi-line attributes (`#[derive(\\n  Foo\\n)]`) are only partially
  handled: each line starting with `#[` is treated as an attribute line.
- Macro invocations like `println!(...)` are filtered out when discovering
  callees by means of an explicit keyword/macro blocklist plus a `!`
  detection check.
- Duplicate function extraction (across recursion) is prevented-ish by tracking
  `(absolute_file_path, definition_start_line_number_zero_indexed)` tuples
  in a shared dedup set to check against.

Python version: 3.9+ (uses PEP 604 `str | None` style unions).
"""

import os
import re
import time
import traceback
from datetime import datetime

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Rust keywords and common macros/builtins that should NOT be treated as
# user-defined function callees when scanning a function body for further
# recursion. This is intentionally conservative: anything in this set is
# skipped during callee discovery.
RUST_NON_CALLABLE_IDENTIFIER_BLOCKLIST_SET: set[str] = {
    # Control-flow keywords
    "if", "else", "while", "for", "loop", "match", "return", "break",
    "continue", "in", "as", "where", "move", "ref", "mut", "let", "const",
    "static", "fn", "impl", "trait", "struct", "enum", "union", "use",
    "mod", "pub", "crate", "super", "self", "Self", "type", "dyn", "async",
    "await", "unsafe", "extern", "true", "false",
    # Common macros (these end with ! so they would normally be filtered,
    # but we add their bare names defensively)
    "println", "print", "eprintln", "eprint", "format", "write", "writeln",
    "vec", "panic", "assert", "assert_eq", "assert_ne", "debug_assert",
    "debug_assert_eq", "debug_assert_ne", "dbg", "todo", "unimplemented",
    "unreachable", "include_str", "include_bytes", "env", "option_env",
    "concat", "stringify", "file", "line", "module_path", "cfg",
    # Common primitive-ish constructors that we don't want to recurse into
    "Some", "None", "Ok", "Err", "Box", "Vec", "String", "Option", "Result",
    "Arc", "Rc", "Mutex", "RwLock", "HashMap", "HashSet", "BTreeMap",
    "BTreeSet",
}


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------

def _generate_iso8601_timestamp_string() -> str:
    """
    Generate a filesystem-safe ISO8601 timestamp string with microsecond
    precision.

    Returns
    -------
    str
        A string of the form `YYYY-MM-DDTHH_MM_SS_mmmmmm` (colons replaced
        with underscores so the value is safe for use in filenames on all
        common operating systems, including Windows).

    Example
    -------
    >>> _generate_iso8601_timestamp_string()
    '2024-01-15T14_32_45_123456'
    """
    # Use local time (naive) per the spec example which does not include a
    # timezone suffix. Replace ':' with '_' for filesystem safety.
    current_datetime_object = datetime.now()
    raw_iso_string = current_datetime_object.isoformat(timespec="microseconds")
    filesystem_safe_iso_string = raw_iso_string.replace(":", "_")
    return filesystem_safe_iso_string


def _log_error_message_to_file_and_terminal(
    error_log_absolute_file_path: str,
    error_message_text: str,
) -> None:
    """
    Append an error message to the run's error log file AND print it to the
    terminal.

    Parameters
    ----------
    error_log_absolute_file_path : str
        Absolute path to the error log text file. The file will be created
        if it does not exist.
    error_message_text : str
        The full error message (typically including a traceback) to record.

    Behavior
    --------
    Never raises. If writing to the log file itself fails, the failure is
    printed to the terminal and otherwise swallowed so that the calling
    extraction process can continue.
    """
    # Always print to terminal first so the user sees the error even if the
    # log file write fails.
    print(f"[function_finder ERROR] {error_message_text}")

    try:
        # Append mode so the log accumulates across many errors in one run.
        with open(error_log_absolute_file_path, "a", encoding="utf-8") as error_log_file_handle:
            error_log_file_handle.write(error_message_text)
            error_log_file_handle.write("\n" + ("-" * 80) + "\n")
    except Exception as inner_log_write_exception:
        # We cannot rely on the log file; print and continue.
        print(
            "[function_finder ERROR] Additionally, failed to write the above "
            f"error to the log file at {error_log_absolute_file_path!r}: "
            f"{inner_log_write_exception!r}"
        )


def _collect_rust_source_files_with_depth_limit(
    root_directory_absolute_path: str,
    maximum_nested_directory_depth: int,
) -> list[str]:
    """
    Walk a directory tree and return absolute paths of all `.rs` files found
    up to a given nested-directory depth.

    Parameters
    ----------
    root_directory_absolute_path : str
        Absolute (or at least valid) path to the directory to search.
    maximum_nested_directory_depth : int
        How many directory levels below the root to descend. A value of 0
        means only files DIRECTLY inside the root directory are considered.
        A value of 2 means root + up to 2 nested subdirectory levels.

    Returns
    -------
    list[str]
        Sorted list of absolute paths to `.rs` files found within the
        depth limit.

    Raises
    ------
    FileNotFoundError
        If `root_directory_absolute_path` does not exist or is not a
        directory.
    """
    if not os.path.isdir(root_directory_absolute_path):
        raise FileNotFoundError(
            f"Cannot collect Rust files: directory does not exist or is not "
            f"a directory: {root_directory_absolute_path!r}"
        )

    # Normalize the root so depth calculations are reliable.
    normalized_root_absolute_path = os.path.abspath(root_directory_absolute_path)
    discovered_rust_file_absolute_paths_list: list[str] = []

    # os.walk yields (current_dir, subdirs, files). We compute the depth of
    # current_dir relative to the root by counting path separators.
    for current_walked_directory_path, _subdirectory_names_list, file_names_in_current_dir in os.walk(
        normalized_root_absolute_path
    ):
        # Compute how deep below root we currently are.
        relative_path_from_root = os.path.relpath(
            current_walked_directory_path, normalized_root_absolute_path
        )
        if relative_path_from_root == ".":
            current_depth_below_root_integer = 0
        else:
            current_depth_below_root_integer = len(
                relative_path_from_root.split(os.sep)
            )

        # Skip directories that are deeper than allowed.
        if current_depth_below_root_integer > maximum_nested_directory_depth:
            continue

        # Collect .rs files at this level.
        for single_file_name_string in file_names_in_current_dir:
            if single_file_name_string.endswith(".rs"):
                full_rust_file_absolute_path = os.path.join(
                    current_walked_directory_path, single_file_name_string
                )
                discovered_rust_file_absolute_paths_list.append(
                    full_rust_file_absolute_path
                )

    discovered_rust_file_absolute_paths_list.sort()
    return discovered_rust_file_absolute_paths_list


def _build_function_definition_detection_regex(function_name_to_find: str) -> "re.Pattern[str]":
    """
    Build a compiled regex that matches a line which begins (after optional
    leading whitespace) with a Rust function definition for the given
    function name.

    Parameters
    ----------
    function_name_to_find : str
        Exact function identifier to search for.

    Returns
    -------
    re.Pattern[str]
        Compiled regex pattern. A match indicates the line starts a function
        definition for `function_name_to_find`.

    Notes
    -----
    Accepts the following common Rust visibility/modifier prefixes before
    `fn`:
      - optional visibility: `pub`, `pub(crate)`, `pub(super)`,
        `pub(in path::to::mod)`
      - optional modifiers (any order, any combination): `async`, `unsafe`,
        `const`, `extern "ABI"`, `default`
    The function name must be followed by either `<` (generic parameters)
    or `(` (parameter list), optionally with whitespace in between.
    """
    # Escape just in case the function name has regex-significant chars
    # (very unlikely for a valid Rust identifier, but defensive).
    escaped_function_name_string = re.escape(function_name_to_find)

    # Visibility (optional): pub OR pub(...)
    visibility_pattern_fragment = r"(?:pub(?:\([^)]*\))?\s+)?"
    # Modifier keywords in any order/combination, each optional.
    modifier_pattern_fragment = (
        r"(?:(?:async|unsafe|const|default)\s+)*"
        r"(?:extern\s+(?:\"[^\"]*\"\s+)?)?"
        r"(?:(?:async|unsafe|const|default)\s+)*"
    )

    full_regex_pattern_string = (
        r"^\s*"
        + visibility_pattern_fragment
        + modifier_pattern_fragment
        + r"fn\s+"
        + escaped_function_name_string
        + r"\s*[<(]"
    )

    return re.compile(full_regex_pattern_string)


def _build_top_level_item_definition_detection_regex() -> "re.Pattern[str]":
    """
    Build a compiled regex that matches a line which begins (after optional
    leading whitespace) with ANY Rust top-level item definition. This is
    used to locate the start of "whatever comes next" after a function, so
    we can determine where a function's extracted region must end.

    Returns
    -------
    re.Pattern[str]
        Compiled regex pattern. A match indicates the line starts a
        top-level item definition of one of these kinds:
          fn, struct, enum, trait, impl, mod, type, union, const, static

    Notes
    -----
    Accepts the same visibility/modifier prefixes as
    `_build_function_definition_detection_regex`:
      - optional visibility: `pub`, `pub(crate)`, `pub(super)`,
        `pub(in path::to::mod)`
      - optional modifiers (any order/combination): `async`, `unsafe`,
        `const`, `extern "ABI"`, `default`

    The matched keyword (`fn`, `struct`, ...) must be followed by
    whitespace, `<`, `(`, `{`, `:`, or end-of-line, so we do not falsely
    match identifiers that merely start with one of these keywords
    (e.g. `function_x`, `structured_thing`).
    """
    # Visibility (optional): pub OR pub(...)
    visibility_pattern_fragment = r"(?:pub(?:\([^)]*\))?\s+)?"
    # Modifier keywords in any order/combination, each optional.
    modifier_pattern_fragment = (
        r"(?:(?:async|unsafe|const|default)\s+)*"
        r"(?:extern\s+(?:\"[^\"]*\"\s+)?)?"
        r"(?:(?:async|unsafe|const|default)\s+)*"
    )
    # The set of item keywords whose appearance marks the start of a new
    # top-level item region.
    item_keyword_alternation_pattern_fragment = (
        r"(?:fn|struct|enum|trait|impl|mod|type|union|const|static)"
    )

    full_regex_pattern_string = (
        r"^\s*"
        + visibility_pattern_fragment
        + modifier_pattern_fragment
        + item_keyword_alternation_pattern_fragment
        + r"(?=\s|<|\(|\{|:|$)"
    )

    return re.compile(full_regex_pattern_string)


# Pre-compile once at module load for reuse across files.
_COMPILED_TOP_LEVEL_ITEM_DEFINITION_REGEX = _build_top_level_item_definition_detection_regex()


def _find_function_definition_line_indices_in_file_lines(
    file_text_lines_list: list[str],
    function_name_to_find: str,
) -> list[int]:
    """
    Scan a list of file lines and return zero-indexed line numbers of every
    line that begins a function definition for the given function name.

    Parameters
    ----------
    file_text_lines_list : list[str]
        The full content of a `.rs` file split into lines (line endings may
        or may not be present; this function does not depend on them).
    function_name_to_find : str
        Exact function name to match.

    Returns
    -------
    list[int]
        Zero-indexed line numbers at which a matching function definition
        starts. Empty list if no matches found.
    """
    compiled_definition_regex_pattern = _build_function_definition_detection_regex(
        function_name_to_find
    )
    matching_line_indices_list: list[int] = []

    for line_index_integer, single_line_string in enumerate(file_text_lines_list):
        if compiled_definition_regex_pattern.match(single_line_string):
            matching_line_indices_list.append(line_index_integer)

    return matching_line_indices_list


def _find_function_top_boundary_line_index(
    file_text_lines_list: list[str],
    function_definition_line_index: int,
) -> int:
    """
    Given the zero-indexed line of a function definition, climb upward
    through the file to include any immediately preceding doc-comment lines
    (`///` or `//!`) and attribute lines (`#[...]`).

    Parameters
    ----------
    file_text_lines_list : list[str]
        The full content of a `.rs` file split into lines.
    function_definition_line_index : int
        Zero-indexed line at which the function definition's `fn name(...)`
        line lives.

    Returns
    -------
    int
        Zero-indexed top boundary line. Always <= function_definition_line_index.

    Behavior
    --------
    Stops climbing as soon as a line is encountered that is NOT one of:
      - a `///` doc-comment line
      - a `//!` inner-doc-comment line
      - an attribute line starting with `#[`
    Blank lines BREAK the climb (they are not included in the extracted
    region), matching the user's stated heuristic.
    """
    top_boundary_line_index_integer = function_definition_line_index

    # Climb upward (toward index 0).
    candidate_line_index_integer = function_definition_line_index - 1
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
            # Anything else (including blank lines) breaks the climb.
            break

    return top_boundary_line_index_integer


def _find_all_top_level_item_region_start_line_indices_in_file(
    file_text_lines_list: list[str],
) -> list[int]:
    """
    Pass 1 of the two-pass extraction strategy.

    Scan an entire `.rs` file's lines and return a sorted list of zero-indexed
    line numbers, each one being the FIRST line of some top-level item's
    extracted region.

    A "top-level item's extracted region" begins at the topmost contiguous
    `///` / `//!` / `#[...]` line immediately above the item's definition
    line, or at the definition line itself if there are no such preceding
    lines.

    Parameters
    ----------
    file_text_lines_list : list[str]
        The full content of a `.rs` file split into lines.

    Returns
    -------
    list[int]
        Sorted, ascending list of zero-indexed region-start line numbers.
        Used by `_find_function_bottom_boundary_line_index_via_next_region_start`
        to determine where a given function's region must end (namely, one
        line before the next region-start).

    Notes
    -----
    Top-level item kinds detected: `fn`, `struct`, `enum`, `trait`, `impl`,
    `mod`, `type`, `union`, `const`, `static`. The set of recognized prefixes
    (visibility / `async` / `unsafe` / `const` / `extern "ABI"` / `default`)
    mirrors that of `_build_function_definition_detection_regex`.

    Climbing upward through doc-comments and attributes reuses
    `_find_function_top_boundary_line_index` so the rule is identical to
    the one used for the target function.
    """
    region_start_line_indices_list: list[int] = []

    for line_index_integer, single_line_string in enumerate(file_text_lines_list):
        if _COMPILED_TOP_LEVEL_ITEM_DEFINITION_REGEX.match(single_line_string):
            # Climb upward through doc-comments and attributes to find the
            # true start of this item's region.
            region_top_line_index_integer = _find_function_top_boundary_line_index(
                file_text_lines_list,
                line_index_integer,
            )
            region_start_line_indices_list.append(region_top_line_index_integer)

    # Sort to be safe; in practice the scan is already in order, but
    # climbing upward could in pathological cases produce a non-monotonic
    # sequence if two items share preceding lines. Sorting and de-duplicating
    # makes the lookup robust.
    region_start_line_indices_list = sorted(set(region_start_line_indices_list))
    return region_start_line_indices_list

# # deprecated version
# def old_find_function_bottom_boundary_line_index_via_brace_counting(
#     file_text_lines_list: list[str],
#     function_definition_line_index: int,
# ) -> int:
#     """
#     Given the zero-indexed line of a function definition, find the
#     zero-indexed line containing the closing `}` of the function body using
#     naive brace counting.

#     Parameters
#     ----------
#     file_text_lines_list : list[str]
#         The full content of a `.rs` file split into lines.
#     function_definition_line_index : int
#         Zero-indexed line at which the function definition's `fn name(...)`
#         line lives.

#     Returns
#     -------
#     int
#         Zero-indexed line of the closing `}`. If no opening brace is found
#         (e.g., a trait method signature ending with `;`), returns
#         `function_definition_line_index` (single-line signature).

#     Limitations
#     -----------
#     Braces inside string literals, character literals, raw strings, or
#     comments will mislead this counter. This is accepted per the project
#     spec. A future improvement could strip strings/comments before counting.
#     """
#     running_brace_balance_integer = 0
#     have_seen_first_opening_brace_flag = False
#     total_file_line_count_integer = len(file_text_lines_list)

#     current_line_index_integer = function_definition_line_index
#     while current_line_index_integer < total_file_line_count_integer:
#         single_line_string = file_text_lines_list[current_line_index_integer]

#         for single_character_in_line in single_line_string:
#             if single_character_in_line == "{":
#                 running_brace_balance_integer += 1
#                 have_seen_first_opening_brace_flag = True
#             elif single_character_in_line == "}":
#                 running_brace_balance_integer -= 1

#             # Once we've seen the body open and the counter returns to zero,
#             # this character closes the function body.
#             if (
#                 have_seen_first_opening_brace_flag
#                 and running_brace_balance_integer == 0
#             ):
#                 return current_line_index_integer

#         # Heuristic fallback: if the signature ends with `;` (e.g., a trait
#         # method declaration without a body) and we never saw `{`, treat
#         # the signature line itself as the end.
#         if (
#             not have_seen_first_opening_brace_flag
#             and single_line_string.rstrip().endswith(";")
#         ):
#             return current_line_index_integer

#         current_line_index_integer += 1

#     # If we fell off the end, return the last line as a safe fallback.
#     return total_file_line_count_integer - 1


def _find_function_bottom_boundary_line_index_via_next_region_start(
    file_text_lines_list: list[str],
    function_definition_line_index: int,
    sorted_region_start_line_indices_list: list[int],
) -> int:
    """
    Pass 2 bottom-finder.

    Given a function's definition line index and the precomputed sorted list
    of region-start line indices for the whole file (from
    `_find_all_top_level_item_region_start_line_indices_in_file`), return
    the zero-indexed last line of this function's extracted region.

    The rule is: the function's region ends one line BEFORE the next
    top-level item's region begins. If there is no following item in the
    file, the region ends at the last line of the file.

    Parameters
    ----------
    file_text_lines_list : list[str]
        The full content of a `.rs` file split into lines.
    function_definition_line_index : int
        Zero-indexed line at which the target function's `fn name(...)`
        line lives. Note: this is the DEFINITION line, not the top of the
        extracted region; we compare against it because doc-comments and
        attributes above the function belong to THIS function, not to the
        previous one.
    sorted_region_start_line_indices_list : list[int]
        Sorted ascending list of region-start line indices for every
        top-level item in the file.

    Returns
    -------
    int
        Zero-indexed line number of the last line of the function's
        extracted region (inclusive).

    Rationale
    ---------
    This replaces brace counting, which was unreliable because braces
    inside string literals, character literals, raw strings, line comments,
    block comments, and macro bodies can desynchronize the counter and
    silently swallow large portions of the file. The "next region start"
    rule avoids parsing the body entirely.
    """
    total_file_line_count_integer = len(file_text_lines_list)

    # Find the smallest region-start line index that is strictly greater
    # than the target function's definition line index.
    next_region_start_line_index_or_none: int | None = None
    for single_region_start_line_index_integer in sorted_region_start_line_indices_list:
        if single_region_start_line_index_integer > function_definition_line_index:
            next_region_start_line_index_or_none = single_region_start_line_index_integer
            break

    if next_region_start_line_index_or_none is None:
        # No following item: this function runs to the end of the file.
        return total_file_line_count_integer - 1

    # Otherwise, stop one line before the next region begins.
    bottom_boundary_line_index_integer = next_region_start_line_index_or_none - 1

    # Defensive clamp: never return a line index below the definition line.
    if bottom_boundary_line_index_integer < function_definition_line_index:
        bottom_boundary_line_index_integer = function_definition_line_index

    return bottom_boundary_line_index_integer


def _extract_callee_function_names_from_body_lines(
    function_body_lines_list: list[str],
) -> set[str]:
    """
    Heuristically extract identifiers that look like callees from a slice
    of function body lines.

    Parameters
    ----------
    function_body_lines_list : list[str]
        Lines belonging to a single extracted function (including its
        signature; non-body lines are harmless to scan).

    Returns
    -------
    set[str]
        Unique candidate callee identifiers. Rust keywords, common macros,
        and common type/enum constructors are filtered out via
        `RUST_NON_CALLABLE_IDENTIFIER_BLOCKLIST_SET`. Macro invocations
        (`identifier!(...)`) are also excluded.
    """
    # Regex: capture an identifier that is immediately followed by '(' OR
    # by '!(' (we will reject the latter as a macro). Also captures the
    # character immediately after the identifier so we can check for '!'.
    identifier_followed_by_paren_or_bang_regex = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)(\s*!?\s*\()"
    )

    discovered_callee_names_set: set[str] = set()

    for single_body_line_string in function_body_lines_list:
        # Quick-and-dirty: strip line comments to reduce false positives.
        # (We do not parse out string literals; that is acknowledged.)
        line_without_trailing_comment = single_body_line_string.split("//", 1)[0]

        for regex_match_object in identifier_followed_by_paren_or_bang_regex.finditer(
            line_without_trailing_comment
        ):
            candidate_identifier_string = regex_match_object.group(1)
            characters_after_identifier_string = regex_match_object.group(2)

            # Reject macro invocations (identifier followed by '!').
            if "!" in characters_after_identifier_string:
                continue

            # Reject blocklisted identifiers (keywords, common macros,
            # common constructors).
            if candidate_identifier_string in RUST_NON_CALLABLE_IDENTIFIER_BLOCKLIST_SET:
                continue

            discovered_callee_names_set.add(candidate_identifier_string)

    return discovered_callee_names_set


def _write_extracted_function_lines_to_unique_output_file(
    output_directory_absolute_path: str,
    function_name_string: str,
    run_timestamp_string: str,
    lines_to_write_list: list[str],
    error_log_absolute_file_path: str,
) -> str:
    """
    Write a slice of file lines (the extracted function plus its preceding
    doc/attribute lines) to a uniquely named output file. If the chosen
    filename already exists, wait briefly and retry with an incrementing
    numeric suffix.

    Parameters
    ----------
    output_directory_absolute_path : str
        Absolute path to the `function_finder_files/` directory.
    function_name_string : str
        The function's name (used as filename prefix).
    run_timestamp_string : str
        ISO8601 timestamp string shared across this run.
    lines_to_write_list : list[str]
        Exact lines (in order) to write to the output file. Line endings
        should already be present in each element (as read from the source).
    error_log_absolute_file_path : str
        Path used to log any unexpected I/O issues.

    Returns
    -------
    str
        Absolute path to the file that was actually written.

    Raises
    ------
    OSError
        If after multiple retries no unique filename could be obtained.
    """
    # Primary candidate filename per spec: {function_name}_{timestamp}.rs
    base_filename_string = f"{function_name_string}_{run_timestamp_string}.rs"
    candidate_full_path_string = os.path.join(
        output_directory_absolute_path, base_filename_string
    )

    maximum_collision_retry_attempts_integer = 50
    collision_retry_attempt_counter_integer = 0

    # On collision: wait briefly then append a numeric suffix.
    while os.path.exists(candidate_full_path_string):
        collision_retry_attempt_counter_integer += 1
        if collision_retry_attempt_counter_integer > maximum_collision_retry_attempts_integer:
            raise OSError(
                "Could not obtain a unique output filename after "
                f"{maximum_collision_retry_attempts_integer} attempts for "
                f"function {function_name_string!r} in directory "
                f"{output_directory_absolute_path!r}."
            )

        # Brief wait so a parallel writer can finish, then retry with a
        # numeric suffix.
        time.sleep(0.01)
        candidate_filename_with_suffix_string = (
            f"{function_name_string}_{run_timestamp_string}"
            f"__dup{collision_retry_attempt_counter_integer}.rs"
        )
        candidate_full_path_string = os.path.join(
            output_directory_absolute_path,
            candidate_filename_with_suffix_string,
        )

    try:
        with open(candidate_full_path_string, "w", encoding="utf-8") as output_file_handle:
            for single_output_line_string in lines_to_write_list:
                output_file_handle.write(single_output_line_string)
                # Ensure each written line ends with a newline; if the
                # source already had one this would double up, so only add
                # if missing.
                if not single_output_line_string.endswith("\n"):
                    output_file_handle.write("\n")
    except Exception as file_write_exception_object:
        # Log and re-raise so the caller can decide what to do.
        full_traceback_text_string = traceback.format_exc()
        _log_error_message_to_file_and_terminal(
            error_log_absolute_file_path,
            (
                f"Failed to write extracted function {function_name_string!r} "
                f"to {candidate_full_path_string!r}: "
                f"{file_write_exception_object!r}\n{full_traceback_text_string}"
            ),
        )
        raise

    return candidate_full_path_string


# ---------------------------------------------------------------------------
# Internal recursive worker that all public entry points share
# ---------------------------------------------------------------------------

def _function_finder_recursive_internal_worker(
    rust_code_dir_absolute_path: str,
    function_name_to_find: str,
    remaining_function_depth_integer: int,
    file_depth_limit_integer: int,
    only_search_this_file_absolute_path_or_none: str | None,
    output_directory_absolute_path: str,
    run_timestamp_string: str,
    error_log_absolute_file_path: str,
    already_found_dedup_set: set[tuple[str, int]],
    produced_output_file_paths_list_accumulator: list[str],
) -> None:
    """
    Internal recursive worker that performs the actual extraction. The
    public `function_finder` wraps this and supplies/initializes the shared
    state (timestamp, error log path, dedup set, produced files list).

    Parameters
    ----------
    rust_code_dir_absolute_path : str
        Directory tree to search.
    function_name_to_find : str
        Exact function name to extract.
    remaining_function_depth_integer : int
        How many more levels of callee-recursion to perform. 0 means do not
        recurse.
    file_depth_limit_integer : int
        How many nested directory levels to descend during file discovery.
    only_search_this_file_absolute_path_or_none : str | None
        If provided, restrict the search to this single file (and disable
        further callee recursion for this invocation).
    output_directory_absolute_path : str
        Absolute path to the `function_finder_files/` output directory.
    run_timestamp_string : str
        Shared ISO8601 timestamp for this run.
    error_log_absolute_file_path : str
        Path to the run-wide error log file.
    already_found_dedup_set : set[tuple[str, int]]
        Shared set of (file_path, definition_line_index) tuples used to
        avoid extracting the same function twice during one run.
    produced_output_file_paths_list_accumulator : list[str]
        Shared mutable list to which every produced file path is appended.

    Returns
    -------
    None
        Side effects only (writes files, mutates the dedup set and the
        accumulator list, logs errors).
    """
    # 1) Determine the set of files to scan.
    try:
        if only_search_this_file_absolute_path_or_none is not None:
            if not os.path.isfile(only_search_this_file_absolute_path_or_none):
                raise FileNotFoundError(
                    "only_search_this_file_path was provided but does not "
                    f"point to a file: {only_search_this_file_absolute_path_or_none!r}"
                )
            files_to_scan_absolute_paths_list = [
                os.path.abspath(only_search_this_file_absolute_path_or_none)
            ]
        else:
            files_to_scan_absolute_paths_list = (
                _collect_rust_source_files_with_depth_limit(
                    rust_code_dir_absolute_path,
                    file_depth_limit_integer,
                )
            )
    except Exception as file_discovery_exception_object:
        full_traceback_text_string = traceback.format_exc()
        _log_error_message_to_file_and_terminal(
            error_log_absolute_file_path,
            (
                "Failed during file discovery for function "
                f"{function_name_to_find!r} in directory "
                f"{rust_code_dir_absolute_path!r}: "
                f"{file_discovery_exception_object!r}\n"
                f"{full_traceback_text_string}"
            ),
        )
        return

    # 2) Scan each file.
    for single_rust_file_absolute_path_string in files_to_scan_absolute_paths_list:
        try:
            with open(
                single_rust_file_absolute_path_string,
                "r",
                encoding="utf-8",
                errors="replace",
            ) as rust_source_file_handle:
                # Keep line endings so we can write them back faithfully.
                file_text_lines_list = rust_source_file_handle.readlines()
        except Exception as file_read_exception_object:
            full_traceback_text_string = traceback.format_exc()
            _log_error_message_to_file_and_terminal(
                error_log_absolute_file_path,
                (
                    f"Failed to read file {single_rust_file_absolute_path_string!r}"
                    f" while searching for {function_name_to_find!r}: "
                    f"{file_read_exception_object!r}\n"
                    f"{full_traceback_text_string}"
                ),
            )
            continue

        # Pass 1 (per file): precompute the sorted region-start line indices
        # for every top-level item in this file. Used to determine where
        # the target function's extracted region must end.
        try:
            sorted_region_start_line_indices_list_for_this_file = (
                _find_all_top_level_item_region_start_line_indices_in_file(
                    file_text_lines_list
                )
            )
        except Exception as pass_one_scan_exception_object:
            full_traceback_text_string = traceback.format_exc()
            _log_error_message_to_file_and_terminal(
                error_log_absolute_file_path,
                (
                    "Failed pass-1 region-start scan in file "
                    f"{single_rust_file_absolute_path_string!r} for "
                    f"{function_name_to_find!r}: "
                    f"{pass_one_scan_exception_object!r}\n"
                    f"{full_traceback_text_string}"
                ),
            )
            continue

        # 3) Find each function definition occurrence in the file.
        try:
            matching_definition_line_indices_list = (
                _find_function_definition_line_indices_in_file_lines(
                    file_text_lines_list, function_name_to_find
                )
            )
        except Exception as regex_scan_exception_object:
            full_traceback_text_string = traceback.format_exc()
            _log_error_message_to_file_and_terminal(
                error_log_absolute_file_path,
                (
                    f"Failed regex scan in file "
                    f"{single_rust_file_absolute_path_string!r} for "
                    f"{function_name_to_find!r}: "
                    f"{regex_scan_exception_object!r}\n"
                    f"{full_traceback_text_string}"
                ),
            )
            continue

        # 4) Extract each occurrence.
        for single_definition_line_index_integer in matching_definition_line_indices_list:
            dedup_key_tuple = (
                single_rust_file_absolute_path_string,
                single_definition_line_index_integer,
            )
            if dedup_key_tuple in already_found_dedup_set:
                # Already extracted this exact occurrence in this run.
                continue
            already_found_dedup_set.add(dedup_key_tuple)

            try:
                top_boundary_line_index_integer = (
                    _find_function_top_boundary_line_index(
                        file_text_lines_list,
                        single_definition_line_index_integer,
                    )
                )
                bottom_boundary_line_index_integer = (
                    _find_function_bottom_boundary_line_index_via_next_region_start(
                        file_text_lines_list,
                        single_definition_line_index_integer,
                        sorted_region_start_line_indices_list_for_this_file,
                    )
                )

                extracted_function_lines_list = file_text_lines_list[
                    top_boundary_line_index_integer : bottom_boundary_line_index_integer + 1
                ]

                # Prepend a small header comment for traceability.
                provenance_header_comment_string = (
                    f"// Extracted by function_finder\n"
                    f"// Source file: {single_rust_file_absolute_path_string}\n"
                    f"// Source lines (1-indexed): "
                    f"{top_boundary_line_index_integer + 1}"
                    f"..{bottom_boundary_line_index_integer + 1}\n"
                    f"// Function name searched: {function_name_to_find}\n"
                    f"\n"
                )
                final_output_lines_list = [provenance_header_comment_string] + extracted_function_lines_list

                written_output_file_path_string = (
                    _write_extracted_function_lines_to_unique_output_file(
                        output_directory_absolute_path,
                        function_name_to_find,
                        run_timestamp_string,
                        final_output_lines_list,
                        error_log_absolute_file_path,
                    )
                )
                produced_output_file_paths_list_accumulator.append(
                    written_output_file_path_string
                )

                # 5) If recursion is requested, discover callees and recurse.
                if remaining_function_depth_integer > 0:
                    try:
                        discovered_callee_names_set = (
                            _extract_callee_function_names_from_body_lines(
                                extracted_function_lines_list
                            )
                        )
                    except Exception as callee_extraction_exception_object:
                        full_traceback_text_string = traceback.format_exc()
                        _log_error_message_to_file_and_terminal(
                            error_log_absolute_file_path,
                            (
                                "Failed to extract callee names from function "
                                f"{function_name_to_find!r} in file "
                                f"{single_rust_file_absolute_path_string!r}: "
                                f"{callee_extraction_exception_object!r}\n"
                                f"{full_traceback_text_string}"
                            ),
                        )
                        discovered_callee_names_set = set()

                    # Do not recurse into the function we just extracted.
                    discovered_callee_names_set.discard(function_name_to_find)

                    for single_callee_name_string in sorted(discovered_callee_names_set):
                        try:
                            _function_finder_recursive_internal_worker(
                                rust_code_dir_absolute_path=rust_code_dir_absolute_path,
                                function_name_to_find=single_callee_name_string,
                                remaining_function_depth_integer=(
                                    remaining_function_depth_integer - 1
                                ),
                                file_depth_limit_integer=file_depth_limit_integer,
                                only_search_this_file_absolute_path_or_none=None,
                                output_directory_absolute_path=output_directory_absolute_path,
                                run_timestamp_string=run_timestamp_string,
                                error_log_absolute_file_path=error_log_absolute_file_path,
                                already_found_dedup_set=already_found_dedup_set,
                                produced_output_file_paths_list_accumulator=(
                                    produced_output_file_paths_list_accumulator
                                ),
                            )
                        except Exception as recursive_call_exception_object:
                            full_traceback_text_string = traceback.format_exc()
                            _log_error_message_to_file_and_terminal(
                                error_log_absolute_file_path,
                                (
                                    "Failure during recursive search for "
                                    f"callee {single_callee_name_string!r} "
                                    f"(from {function_name_to_find!r}): "
                                    f"{recursive_call_exception_object!r}\n"
                                    f"{full_traceback_text_string}"
                                ),
                            )

            except Exception as per_occurrence_exception_object:
                full_traceback_text_string = traceback.format_exc()
                _log_error_message_to_file_and_terminal(
                    error_log_absolute_file_path,
                    (
                        f"Failed to extract occurrence of "
                        f"{function_name_to_find!r} at line "
                        f"{single_definition_line_index_integer + 1} in file "
                        f"{single_rust_file_absolute_path_string!r}: "
                        f"{per_occurrence_exception_object!r}\n"
                        f"{full_traceback_text_string}"
                    ),
                )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def function_finder(
    rust_code_dir_path: str,
    function_name_to_find: str,
    function_depth: int = 0,
    file_depth: int = 2,
    only_search_this_file_path: str | None = None,
    output_dir: str | None = None,
) -> list[str]:
    """
    Locate and extract one or more Rust functions by name from a directory
    of `.rs` source files, writing each extracted occurrence to its own
    `.rs` file inside `function_finder_files/`.

    Parameters
    ----------
    rust_code_dir_path : str
        Path to the directory containing Rust source files to search.
    function_name_to_find : str
        The exact function name (Rust identifier) to locate. All matches
        across all scanned files are extracted; collisions are accepted
        ("if you search for all instances of X, you get all instances of X").
    function_depth : int, default 0
        How many levels of callee-recursion to perform. If 0, only the
        target function is extracted. If 1, also extract every function the
        target calls. If 2, also extract every function THOSE call. Etc.
    file_depth : int, default 2
        Maximum nested-directory depth to descend below `rust_code_dir_path`
        when discovering `.rs` files. 0 = only files directly in the root.
    only_search_this_file_path : str | None, default None
        If provided, restrict the initial search to this single file
        (ignores `file_depth` for the initial search). Note: recursive
        callee searches (when `function_depth > 0`) still search the whole
        `rust_code_dir_path` tree at `file_depth`, because callees may live
        in other files.
    output_dir : str | None, default None
        Directory in which the `function_finder_files/` subdirectory will
        be created. If None, the current working directory is used.

    Returns
    -------
    list[str]
        Absolute paths of ALL files produced by this run: every extracted
        function file plus the run's error log file (the error log file is
        included even if empty).

    Output files
    ------------
    Files are written under:
        {output_dir or cwd}/function_finder_files/

    With names:
        {function_name}_{ISO8601_timestamp}.rs
        fn_finder_error_log_{ISO8601_timestamp}.txt

    Example
    -------
    >>> produced_file_paths_list = function_finder(
    ...     rust_code_dir_path="./my_rust_project/src",
    ...     function_name_to_find="parse_config",
    ...     function_depth=1,
    ...     file_depth=3,
    ... )
    >>> for single_produced_file_path in produced_file_paths_list:
    ...     print(single_produced_file_path)

    Behavior on errors
    ------------------
    Any per-file or per-function error is logged (to the error log file and
    to the terminal) and the run continues. The error log file path is
    always included in the returned list.
    """
    # --- 0. Validate inputs early with clear error messages ----------------
    if not isinstance(rust_code_dir_path, str) or not rust_code_dir_path:
        raise ValueError(
            "function_finder requires `rust_code_dir_path` to be a non-empty "
            f"string; got {rust_code_dir_path!r}."
        )
    if not isinstance(function_name_to_find, str) or not function_name_to_find:
        raise ValueError(
            "function_finder requires `function_name_to_find` to be a "
            f"non-empty string; got {function_name_to_find!r}."
        )
    if not isinstance(function_depth, int) or function_depth < 0:
        raise ValueError(
            "function_finder requires `function_depth` to be a non-negative "
            f"integer; got {function_depth!r}."
        )
    if not isinstance(file_depth, int) or file_depth < 0:
        raise ValueError(
            "function_finder requires `file_depth` to be a non-negative "
            f"integer; got {file_depth!r}."
        )

    if not os.path.isdir(rust_code_dir_path):
        raise FileNotFoundError(
            "function_finder: `rust_code_dir_path` does not exist or is not "
            f"a directory: {rust_code_dir_path!r}"
        )

    # --- 1. Resolve output directory and create function_finder_files/ ----
    effective_output_root_directory_string = (
        output_dir if output_dir is not None else os.getcwd()
    )
    if not os.path.isdir(effective_output_root_directory_string):
        raise FileNotFoundError(
            "function_finder: `output_dir` does not exist or is not a "
            f"directory: {effective_output_root_directory_string!r}"
        )

    function_finder_output_subdirectory_absolute_path = os.path.abspath(
        os.path.join(
            effective_output_root_directory_string, "function_finder_files"
        )
    )
    try:
        os.makedirs(
            function_finder_output_subdirectory_absolute_path, exist_ok=True
        )
    except Exception as output_dir_creation_exception_object:
        full_traceback_text_string = traceback.format_exc()
        raise OSError(
            "function_finder: failed to create output directory "
            f"{function_finder_output_subdirectory_absolute_path!r}: "
            f"{output_dir_creation_exception_object!r}\n"
            f"{full_traceback_text_string}"
        ) from output_dir_creation_exception_object

    # --- 2. Establish run-wide shared state -------------------------------
    run_timestamp_string = _generate_iso8601_timestamp_string()
    error_log_absolute_file_path = os.path.join(
        function_finder_output_subdirectory_absolute_path,
        f"fn_finder_error_log_{run_timestamp_string}.txt",
    )
    # Touch the error log file so it always exists in the returned list.
    try:
        with open(error_log_absolute_file_path, "w", encoding="utf-8") as initial_log_file_handle:
            initial_log_file_handle.write(
                f"function_finder error log\n"
                f"Timestamp: {run_timestamp_string}\n"
                f"rust_code_dir_path: {rust_code_dir_path!r}\n"
                f"function_name_to_find: {function_name_to_find!r}\n"
                f"function_depth: {function_depth}\n"
                f"file_depth: {file_depth}\n"
                f"only_search_this_file_path: {only_search_this_file_path!r}\n"
                f"output_dir: {output_dir!r}\n"
                f"{'-' * 80}\n"
            )
    except Exception as log_init_exception_object:
        # If we cannot even initialize the log, print and continue; the
        # logger helper will keep trying.
        print(
            "[function_finder ERROR] Could not initialize error log file at "
            f"{error_log_absolute_file_path!r}: {log_init_exception_object!r}"
        )

    already_found_dedup_set: set[tuple[str, int]] = set()
    produced_output_file_paths_list_accumulator: list[str] = []

    # --- 3. Kick off the recursive worker ---------------------------------
    try:
        _function_finder_recursive_internal_worker(
            rust_code_dir_absolute_path=os.path.abspath(rust_code_dir_path),
            function_name_to_find=function_name_to_find,
            remaining_function_depth_integer=function_depth,
            file_depth_limit_integer=file_depth,
            only_search_this_file_absolute_path_or_none=(
                only_search_this_file_path
            ),
            output_directory_absolute_path=function_finder_output_subdirectory_absolute_path,
            run_timestamp_string=run_timestamp_string,
            error_log_absolute_file_path=error_log_absolute_file_path,
            already_found_dedup_set=already_found_dedup_set,
            produced_output_file_paths_list_accumulator=(
                produced_output_file_paths_list_accumulator
            ),
        )
    except Exception as top_level_exception_object:
        # The internal worker should not raise, but defensively log if it does.
        full_traceback_text_string = traceback.format_exc()
        _log_error_message_to_file_and_terminal(
            error_log_absolute_file_path,
            (
                "Unexpected top-level exception in function_finder for "
                f"{function_name_to_find!r}: {top_level_exception_object!r}\n"
                f"{full_traceback_text_string}"
            ),
        )

    # --- 4. Always include the error log in the returned list -------------
    produced_output_file_paths_list_accumulator.append(error_log_absolute_file_path)
    return produced_output_file_paths_list_accumulator


def flatten_finder(dir_path: str) -> str:
    """
    Concatenate every `.rs` file directly inside `dir_path` into a single
    flat `.rs` file written to the current working directory.

    Parameters
    ----------
    dir_path : str
        Path to a directory containing `.rs` files (typically the
        `function_finder_files/` directory produced by `function_finder`).
        Only `.rs` files DIRECTLY in this directory are included; nested
        subdirectories are NOT traversed.

    Returns
    -------
    str
        Absolute path to the produced flat file. The file is named
        `flat_functions_{ISO8601_timestamp}.rs` and is placed in the
        current working directory.

    Behavior
    --------
    Each source file's contents are preceded by a header comment of the
    form:
        // ===== source: {original_filename} =====
    so that the boundaries between concatenated files remain visible.

    Error handling
    --------------
    Failures to read individual source files are logged to the terminal
    and skipped; the flat output file is still produced (possibly partial).
    A failure to create the output file itself is raised.
    """
    # --- 0. Validate inputs ------------------------------------------------
    if not isinstance(dir_path, str) or not dir_path:
        raise ValueError(
            "flatten_finder requires `dir_path` to be a non-empty string; "
            f"got {dir_path!r}."
        )
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(
            "flatten_finder: `dir_path` does not exist or is not a "
            f"directory: {dir_path!r}"
        )

    # --- 1. Discover .rs files directly inside dir_path -------------------
    try:
        all_entries_in_directory_list = os.listdir(dir_path)
    except Exception as listdir_exception_object:
        full_traceback_text_string = traceback.format_exc()
        raise OSError(
            f"flatten_finder: failed to list directory {dir_path!r}: "
            f"{listdir_exception_object!r}\n{full_traceback_text_string}"
        ) from listdir_exception_object

    rust_source_file_absolute_paths_list: list[str] = []
    for single_entry_name_string in sorted(all_entries_in_directory_list):
        candidate_full_path_string = os.path.join(dir_path, single_entry_name_string)
        if (
            os.path.isfile(candidate_full_path_string)
            and single_entry_name_string.endswith(".rs")
        ):
            rust_source_file_absolute_paths_list.append(
                os.path.abspath(candidate_full_path_string)
            )

    # --- 2. Build the flat output filename --------------------------------
    flatten_run_timestamp_string = _generate_iso8601_timestamp_string()
    flat_output_filename_string = (
        f"flat_functions_{flatten_run_timestamp_string}.rs"
    )
    flat_output_absolute_file_path = os.path.abspath(
        os.path.join(os.getcwd(), flat_output_filename_string)
    )

    # --- 3. Write the concatenated content --------------------------------
    try:
        with open(
            flat_output_absolute_file_path, "w", encoding="utf-8"
        ) as flat_output_file_handle:
            # Top-of-file header for the flat output.
            flat_output_file_handle.write(
                f"// flatten_finder output\n"
                f"// Source directory: {os.path.abspath(dir_path)}\n"
                f"// Timestamp: {flatten_run_timestamp_string}\n"
                f"// Number of source .rs files concatenated: "
                f"{len(rust_source_file_absolute_paths_list)}\n"
                f"\n"
            )

            for single_source_rust_file_absolute_path_string in rust_source_file_absolute_paths_list:
                section_header_comment_string = (
                    f"// ===== source: "
                    f"{os.path.basename(single_source_rust_file_absolute_path_string)} "
                    f"=====\n"
                )
                flat_output_file_handle.write(section_header_comment_string)

                try:
                    with open(
                        single_source_rust_file_absolute_path_string,
                        "r",
                        encoding="utf-8",
                        errors="replace",
                    ) as single_source_file_handle:
                        flat_output_file_handle.write(
                            single_source_file_handle.read()
                        )
                    flat_output_file_handle.write("\n")
                except Exception as per_file_read_exception_object:
                    full_traceback_text_string = traceback.format_exc()
                    # Log to terminal; embed a notice in the flat output.
                    print(
                        f"[flatten_finder ERROR] Failed to read "
                        f"{single_source_rust_file_absolute_path_string!r}: "
                        f"{per_file_read_exception_object!r}\n"
                        f"{full_traceback_text_string}"
                    )
                    flat_output_file_handle.write(
                        f"// [flatten_finder ERROR] Could not read this "
                        f"source file: "
                        f"{single_source_rust_file_absolute_path_string}\n"
                        f"// Reason: {per_file_read_exception_object!r}\n\n"
                    )
    except Exception as flat_write_exception_object:
        full_traceback_text_string = traceback.format_exc()
        raise OSError(
            "flatten_finder: failed to write flat output file "
            f"{flat_output_absolute_file_path!r}: "
            f"{flat_write_exception_object!r}\n"
            f"{full_traceback_text_string}"
        ) from flat_write_exception_object

    return flat_output_absolute_file_path


# ---------
#  CLI Q&A
# ---------
if __name__ == "__main__":
    # Q&A for user cli

    print(
        "function_finder module. Import and call:\n"
        "    from function_finder import function_finder, flatten_finder\n"
        "    produced_file_paths_list = function_finder(\n"
        "        rust_code_dir_path='./path/to/rust/src',\n"
        "        function_name_to_find='my_function',\n"
        "        function_depth=0,\n"
        "        file_depth=2,\n"
        "        only_search_this_file_path=None,\n"
        "        output_dir=None,\n"
        "    )\n"
        "    flat_path_string = flatten_finder('./function_finder_files')\n"
    )

    print("\n\nFunction Finder (Rust) Q&A\n")

    rust_code_dir_path = input("What is rust_code_dir_path? (e.g. src/\n > ")
    function_name_to_find = input("What is function_name_to_find?\n > ")

    function_depth = input("(optional) What is function_depth? (default 0)\n > ")

    file_depth = input("(optional) What is file_depth? (defualt 2)\n > ")

    only_search_this_file_path = input("(optional) What is only_search_this_file_path?\n > ")

    output_dir = input("(optional) What is output_dir?\n > ")

    if function_depth.strip() == "":
        function_depth = 0

    if file_depth.strip() == "":
        file_depth = 2

    if only_search_this_file_path.strip() == "":
        only_search_this_file_path = None

    if output_dir.strip() == "":
        output_dir = None

    produced_file_paths_list = function_finder(
        rust_code_dir_path=rust_code_dir_path,
        function_name_to_find=function_name_to_find,
        function_depth=int(function_depth),
        file_depth=int(file_depth),
        only_search_this_file_path=only_search_this_file_path,
        output_dir=output_dir,
    )

    flatten_or_not = input("Do you want a flat file?    (y)es / (n)o\n > ")

    if str(flatten_or_not).lower().strip() in ["y", "yes", "true", "ok", "flat"]:
        flat_path_string = flatten_finder('./function_finder_files')
        print(flat_path_string)

    print("\nFind Ok!\n")
