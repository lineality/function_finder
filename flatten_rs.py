import os
import traceback
from datetime import datetime


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


# -----------------------------------------------------------------------
if __name__ == "__main__":

    print("\n\nFile flattener (Rust) Q&A\n")

    dir_of_rs_files_to_flatten = input("What is dir path of rs files to flatten?")

    produced_file_paths_list = flatten_finder(
        dir_of_rs_files_to_flatten,
    )

    print(produced_file_paths_list)

    print("\nFlatland, Ok!\n")
