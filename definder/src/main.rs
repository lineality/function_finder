/// ============================================================================
/// Module: function_finder (definder binary)
/// ============================================================================
///
/// Project context: `definder` is a command-line tool that takes the name of
/// a Rust function definition and searches a single Rust source file for a
/// top-level `fn <name>` signature, then extracts and prints the entire
/// function body (matched by brace depth) to the terminal, one character at
/// a time, without heap allocation in the extraction or output paths.
///
/// Current scope (MVP): operates on exactly one file path given on the
/// command line. Recursive directory search is a stated future goal; the
/// error codes `DirectoryWalkFailed` and `OutputDirectoryCreationFailed`
/// are RESERVED for that future work and are not produced by any code path
/// in this file today. Do not remove them if adding directory-walk support
/// later — the codes must remain append-only per the project error-code
/// policy.
///
/// Adheres to the production-release framework rules:
/// - No heap allocation in production error/output paths.
/// - Terse 2-byte fieldless enum error codes (`FnFinderError`).
/// - Gated diagnostics (`#[cfg(debug_assertions)]`,
///   `#[cfg(all(debug_assertions, not(test)))]`).
/// - Explicit error checking (no `?` operator).
/// - Character-by-character terminal output using stack buffers.
/// ============================================================================
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

/// Project-wide unique error codes for the function_finder module.
///
/// - `Copy`, `Clone`, `PartialEq`, `Eq`: trivially copyable 2-byte value.
/// - `#[repr(u16)]`: forces exact 2-byte size where the discriminant IS the code.
/// - Append-only policy: numbers are never reused or renumbered.
///
/// Error-Code-Table:
/// - 101-199: input validation
/// - 200-299: file / io operations
/// - 300-399: internal invariants (parser / buffer)
#[cfg_attr(any(test, debug_assertions), derive(Debug))]
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u16)]
pub enum FnFinderError {
    // --- Input validation (expected failures on untrusted/external input) ---
    PathIsEmpty = 101,
    FunctionNameIsEmpty = 102,
    TargetFileNotFound = 103,
    /// Reserved for future recursive directory-search support. Not currently produced.
    OutputDirectoryCreationFailed = 104,
    FunctionSignatureNotFound = 105,
    SourceFileTooLargeForStackBuffer = 106,
    /// The located candidate matched "fn <name>" byte-for-byte, but the
    /// following byte is not a valid Rust function-signature continuation
    /// (whitespace or `(`), meaning the match is a longer identifier that
    /// merely starts with the requested name (e.g. searching for
    /// `target_fn` must not match `fn target_fn_helper`).
    FunctionNameIsPrefixOfLongerIdentifier = 107,

    // --- File / IO operations (expected failures; mapped from io::ErrorKind) ---
    FileOpenFailed = 200,
    FileReadFailed = 201,
    FileWriteFailed = 202,
    /// Reserved for future recursive directory-search support. Not currently produced.
    DirectoryWalkFailed = 203,
    FileMetadataUnavailable = 204,

    // --- Internal invariants ("cannot happen" in a correct, uncorrupted run) ---
    BraceStackUnderflow = 300,
    BraceStackUnterminated = 301,
    TerminalWriteFailed = 302,
    SearchPrefixBufferOverflow = 303,
    /// Internal invariant: the backward doc-comment scan computed a start
    /// index that is not a valid boundary within `file_bytes` (e.g. because
    /// `start_idx`, previously validated by `locate_function_signature_start`,
    /// was corrupted between that call and this one). Cannot happen in a
    /// correct, uncorrupted run.
    DocCommentScanIndexOutOfBounds = 304,
}

impl FnFinderError {
    /// Returns the terse numeric error code in all build modes. No heap.
    pub const fn code(self) -> u16 {
        self as u16
    }

    /// Explicit, exhaustive classification of which error codes represent
    /// transient conditions worth retrying. Deliberately not inferred from
    /// numeric ranges (see framework Section 3).
    pub const fn is_retryable(self) -> bool {
        match self {
            FnFinderError::FileOpenFailed
            | FnFinderError::FileReadFailed
            | FnFinderError::FileWriteFailed
            | FnFinderError::FileMetadataUnavailable => true,

            FnFinderError::PathIsEmpty
            | FnFinderError::FunctionNameIsEmpty
            | FnFinderError::TargetFileNotFound
            | FnFinderError::OutputDirectoryCreationFailed
            | FnFinderError::FunctionSignatureNotFound
            | FnFinderError::SourceFileTooLargeForStackBuffer
            | FnFinderError::FunctionNameIsPrefixOfLongerIdentifier
            | FnFinderError::DirectoryWalkFailed
            | FnFinderError::BraceStackUnderflow
            | FnFinderError::BraceStackUnterminated
            | FnFinderError::TerminalWriteFailed
            | FnFinderError::SearchPrefixBufferOverflow => false,
            FnFinderError::DocCommentScanIndexOutOfBounds => false,
        }
    }

    /// Maps a standard library `io::ErrorKind` to a project error code.
    /// The originating `io::Error` (and any path/PII text inside it) is
    /// dropped at the call site; only the discriminant survives here.
    ///
    /// `io::ErrorKind` is `#[non_exhaustive]`, so the wildcard arm is
    /// required by the compiler. Everything routed to the wildcard is a
    /// deliberate "treat as FileOpenFailed, non-retryable-by-default-shape"
    /// policy decision and should be re-audited on Rust version upgrades.
    pub const fn from_io_kind(kind: io::ErrorKind) -> Self {
        match kind {
            io::ErrorKind::NotFound => FnFinderError::TargetFileNotFound,
            io::ErrorKind::Interrupted | io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut => {
                FnFinderError::FileOpenFailed
            }
            _ => FnFinderError::FileOpenFailed,
        }
    }
}

/// Human-readable display implementation for debug and test builds ONLY.
/// Compiled out of production-release builds entirely; no error text is
/// ever embedded in a production-release binary.
#[cfg(debug_assertions)]
impl std::fmt::Display for FnFinderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            FnFinderError::PathIsEmpty => "FNFINDER-101: path string is empty",
            FnFinderError::FunctionNameIsEmpty => "FNFINDER-102: function name string is empty",
            FnFinderError::TargetFileNotFound => "FNFINDER-103: target file does not exist",
            FnFinderError::OutputDirectoryCreationFailed => {
                "FNFINDER-104: failed to create output directory (reserved, unused)"
            }
            FnFinderError::FunctionSignatureNotFound => {
                "FNFINDER-105: no matching function signature found in file"
            }
            FnFinderError::SourceFileTooLargeForStackBuffer => {
                "FNFINDER-106: source file exceeds the fixed stack buffer capacity"
            }
            FnFinderError::FunctionNameIsPrefixOfLongerIdentifier => {
                "FNFINDER-107: matched name is a prefix of a longer identifier"
            }
            FnFinderError::FileOpenFailed => "FNFINDER-200: failed to open file",
            FnFinderError::FileReadFailed => "FNFINDER-201: failed to read file bytes",
            FnFinderError::FileWriteFailed => "FNFINDER-202: failed to write output file",
            FnFinderError::DirectoryWalkFailed => {
                "FNFINDER-203: failed during directory traversal (reserved, unused)"
            }
            FnFinderError::FileMetadataUnavailable => {
                "FNFINDER-204: failed to read file metadata (size check)"
            }
            FnFinderError::BraceStackUnderflow => {
                "FNFINDER-300: encountered closing brace with no matching open brace"
            }
            FnFinderError::BraceStackUnterminated => {
                "FNFINDER-301: function body never reached balanced closing brace"
            }
            FnFinderError::TerminalWriteFailed => "FNFINDER-302: failed writing to terminal handle",
            FnFinderError::SearchPrefixBufferOverflow => {
                "FNFINDER-303: search-prefix stack buffer capacity exceeded"
            }
            FnFinderError::DocCommentScanIndexOutOfBounds => {
                "FNFINDER-304: doc-comment backward scan produced an out-of-bounds index"
            }
        };
        write!(f, "{}", text)
    }
}

/// Specialized Result alias bound to `FnFinderError`.
pub type Result<T> = std::result::Result<T, FnFinderError>;

/// Fixed upper bound on how large a source file this tool will search.
/// Chosen to keep the read buffer stack-allocated (no heap). Files larger
/// than this are explicitly rejected (`SourceFileTooLargeForStackBuffer`)
/// rather than silently truncated.
const MAX_SOURCE_FILE_BYTES: usize = 65536;

/// Fixed upper bound on the length of the "fn <name>" search prefix.
const MAX_SEARCH_PREFIX_BYTES: usize = 128;

// /// Renders a u16 integer to decimal ASCII inside a fixed stack buffer,
// /// with no heap allocation. Available for any future terminal/log surface
// /// that needs to render an `FnFinderError::code()` value in production.
// fn u16_to_decimal(mut value: u16, buf: &mut [u8; 5]) -> &str {
//     let mut start = buf.len();
//     if value == 0 {
//         start -= 1;
//         buf[start] = b'0';
//     } else {
//         while value > 0 && start > 0 {
//             start -= 1;
//             buf[start] = b'0' + (value % 10) as u8;
//             value /= 10;
//         }
//     }
//     match std::str::from_utf8(&buf[start..]) {
//         Ok(s) => s,
//         Err(_) => "?",
//     }
// }

/// Input validation for the two arguments to `finder_rs_for_terminal`.
///
/// Category: input validation (expected failure on caller-provided data).
fn validate_finder_inputs(rust_file_path: &Path, function_name_to_find: &str) -> Result<()> {
    if function_name_to_find.is_empty() {
        #[cfg(debug_assertions)]
        eprintln!("FNFINDER-102: validate_finder_inputs: function name is empty");
        return Err(FnFinderError::FunctionNameIsEmpty);
    }

    if rust_file_path.as_os_str().is_empty() {
        #[cfg(debug_assertions)]
        eprintln!("FNFINDER-101: validate_finder_inputs: file path is empty");
        return Err(FnFinderError::PathIsEmpty);
    }

    if !rust_file_path.exists() {
        #[cfg(debug_assertions)]
        eprintln!("FNFINDER-103: validate_finder_inputs: file not found");
        return Err(FnFinderError::TargetFileNotFound);
    }

    Ok(())
}

/// Opens the given file and reads its entire contents into the caller's
/// fixed-size stack buffer, returning the number of bytes read.
///
/// Category: input validation for size-checking (expected: files vary in
/// size); io-boundary handling for open/metadata/read (expected: any file
/// operation can fail at any time due to permissions, disk state, etc.).
fn read_file_into_stack_buffer(
    rust_file_path: &Path,
    stack_buffer: &mut [u8; MAX_SOURCE_FILE_BYTES],
) -> Result<usize> {
    let mut file_handle = match File::open(rust_file_path) {
        Ok(file) => file,
        Err(detail) => {
            #[cfg(debug_assertions)]
            eprintln!("FNFINDER-200: read_file_into_stack_buffer open: {}", detail);
            return Err(FnFinderError::from_io_kind(detail.kind()));
        }
    };

    let metadata = match file_handle.metadata() {
        Ok(m) => m,
        Err(_detail) => {
            #[cfg(debug_assertions)]
            eprintln!(
                "FNFINDER-204: read_file_into_stack_buffer metadata: {}",
                _detail
            );
            return Err(FnFinderError::FileMetadataUnavailable);
        }
    };

    if metadata.len() as usize > stack_buffer.len() {
        #[cfg(debug_assertions)]
        eprintln!(
            "FNFINDER-106: read_file_into_stack_buffer: file size {} exceeds buffer capacity {}",
            metadata.len(),
            stack_buffer.len()
        );
        return Err(FnFinderError::SourceFileTooLargeForStackBuffer);
    }

    match file_handle.read(stack_buffer) {
        Ok(bytes_read) => Ok(bytes_read),
        Err(detail) => {
            #[cfg(debug_assertions)]
            eprintln!("FNFINDER-201: read_file_into_stack_buffer read: {}", detail);
            Err(FnFinderError::from_io_kind(detail.kind()))
        }
    }
}
/// Searches `file_bytes` for the first occurrence of the byte sequence
/// `"fn " + function_name_to_find`, then confirms the byte immediately
/// following the matched name is a valid function-signature continuation
/// (`(`, space, or tab) rather than another identifier character. This
/// prevents a search for `target_fn` from matching a longer identifier
/// such as `target_fn_helper`.
///
/// Returns `Ok(None)` (not an error) when no valid match is found anywhere
/// in the file, since "the function is not present in this file" is a
/// normal, expected outcome of a search, not a case-handling failure.
///
/// Category: input validation. Both the search-prefix length bound and the
/// following-byte continuation check are expected, non-adversarial
/// conditions that arise from ordinary source file content (e.g. a
/// project that happens to contain a longer, similarly-named function).
/// This function performs no I/O and allocates no heap memory.
fn locate_function_signature_start(
    file_bytes: &[u8],
    function_name_to_find: &str,
) -> Result<Option<usize>> {
    let prefix_len = 3usize
        .checked_add(function_name_to_find.len())
        .unwrap_or(usize::MAX);

    if prefix_len > MAX_SEARCH_PREFIX_BYTES {
        #[cfg(debug_assertions)]
        eprintln!(
            "FNFINDER-303: locate_function_signature_start: prefix length {} exceeds buffer capacity {}",
            prefix_len, MAX_SEARCH_PREFIX_BYTES
        );
        return Err(FnFinderError::SearchPrefixBufferOverflow);
    }

    let mut search_prefix_stack = [0u8; MAX_SEARCH_PREFIX_BYTES];
    let mut cursor = 0usize;
    for b in b"fn " {
        search_prefix_stack[cursor] = *b;
        cursor += 1;
    }
    for b in function_name_to_find.bytes() {
        search_prefix_stack[cursor] = b;
        cursor += 1;
    }
    let target_signature = &search_prefix_stack[..prefix_len];

    if file_bytes.len() < prefix_len {
        return Ok(None);
    }

    // --- Check: does any candidate match continue as another identifier byte? ---
    // Input validation — an ordinary, non-adversarial source file can
    // legitimately contain a longer identifier that begins with the exact
    // bytes being searched for (e.g. `target_fn_helper` when searching for
    // `target_fn`); this is expected, not an internal invariant.
    let mut found_prefix_match_with_bad_continuation = false;

    let mut i = 0usize;
    let last_start = file_bytes.len() - prefix_len;
    while i <= last_start {
        if &file_bytes[i..i + prefix_len] == target_signature {
            let next_byte_index = i + prefix_len;
            let continuation_is_valid = match file_bytes.get(next_byte_index) {
                Some(b'(') | Some(b' ') | Some(b'\t') => true,
                Some(_) => false,
                // End of file immediately after the name: not a valid
                // function signature (no body, no parameter list), but
                // also not a "longer identifier" — treat as no match here
                // and let the caller's absence of a balanced body surface
                // the appropriate downstream error instead.
                None => true,
            };

            if continuation_is_valid {
                return Ok(Some(i));
            }

            found_prefix_match_with_bad_continuation = true;
        }
        i += 1;
    }

    if found_prefix_match_with_bad_continuation {
        #[cfg(debug_assertions)]
        eprintln!(
            "FNFINDER-107: locate_function_signature_start: only prefix-of-longer-identifier matches found"
        );
        return Err(FnFinderError::FunctionNameIsPrefixOfLongerIdentifier);
    }

    Ok(None)
}

/// Scans backward from `signature_start_idx` over any contiguous block of
/// `///` doc-comment lines that immediately precede a function signature,
/// and returns the byte index at which extraction should begin so that
/// those doc-comment lines are included in the extracted function text.
///
/// Project context: `definder` extracts "the entire function body" for
/// display; a function's `///` doc-comments are, by Rust convention, part
/// of that function's definition as read by a human or by `rustdoc`, so
/// omitting them silently would contradict the tool's stated purpose.
///
/// Scope (explicitly bounded, not a general doc-comment parser):
/// - Only bare `///` inner line doc-comments are recognized.
/// - `//!` module-level doc-comments, `/** */` block comments, and
///   `#[attribute]` lines are NOT treated as part of the doc-comment block
///   and are left in place (scanning stops there); supporting them is a
///   separate, not-yet-implemented feature.
/// - A blank or whitespace-only line breaks the contiguous block (matches
///   standard Rust convention that a blank line separates a doc-comment
///   from the item it documents), and scanning stops there.
///
/// Algorithm: repeatedly identify "the line immediately above the current
/// boundary," classify it, and either extend the boundary upward (if it is
/// a `///` line) or stop (otherwise). This checks each preceding line one
/// at a time, from the boundary upward, until a non-`///` line or the
/// start of the file is reached — there is no separate first-iteration
/// special case.
///
/// Category: internal invariant. `signature_start_idx` is expected to have
/// already been validated as a real match location by
/// `locate_function_signature_start`; an index inconsistent with
/// `file_bytes` at this point indicates corruption between calls, not an
/// expected input-validation case, so the full debug_assert + terse-return
/// shape is used.
fn find_doc_comment_block_start(file_bytes: &[u8], signature_start_idx: usize) -> Result<usize> {
    if signature_start_idx > file_bytes.len() {
        #[cfg(all(debug_assertions, not(test)))]
        debug_assert!(
            false,
            "FNFINDER-304: find_doc_comment_block_start: start index {} exceeds file length {}",
            signature_start_idx,
            file_bytes.len()
        );

        #[cfg(debug_assertions)]
        eprintln!(
            "FNFINDER-304: find_doc_comment_block_start: start index {} exceeds file length {}",
            signature_start_idx,
            file_bytes.len()
        );

        return Err(FnFinderError::DocCommentScanIndexOutOfBounds);
    }

    // `boundary` always means: "everything from `boundary` onward, up to
    // and including the signature, is confirmed to be either the
    // signature itself or part of the contiguous doc-comment block."
    // It only ever moves backward (toward index 0).
    let mut boundary = signature_start_idx;

    loop {
        if boundary == 0 {
            break;
        }

        // Identify the line immediately above `boundary`: it ends just
        // before `boundary` (skipping the newline byte at `boundary - 1`,
        // if present) and starts just after the newline before that.
        let mut this_line_end_exclusive = boundary;
        if this_line_end_exclusive > 0 && file_bytes[this_line_end_exclusive - 1] == b'\n' {
            this_line_end_exclusive -= 1;
        } else {
            // `boundary` does not sit immediately after a newline, meaning
            // there is no complete "previous line" above it to examine
            // (this occurs when signature_start_idx is on the file's
            // first line). Stop.
            break;
        }

        let mut this_line_start = this_line_end_exclusive;
        while this_line_start > 0 && file_bytes[this_line_start - 1] != b'\n' {
            this_line_start -= 1;
        }

        let candidate_line = &file_bytes[this_line_start..this_line_end_exclusive];
        let trimmed_candidate = trim_ascii_whitespace(candidate_line);

        if trimmed_candidate.starts_with(b"///") {
            boundary = this_line_start;
            continue;
        }

        // Not a `///` line (blank line, other content, or anything else):
        // the contiguous doc-comment block does not extend further back.
        break;
    }

    Ok(boundary)
}

/// Trims leading and trailing ASCII whitespace (space, tab, carriage
/// return, newline) from a byte slice without allocating. Used only for
/// the doc-comment-line classification check above.
fn trim_ascii_whitespace(bytes: &[u8]) -> &[u8] {
    let mut start = 0usize;
    let mut end = bytes.len();

    while start < end && matches!(bytes[start], b' ' | b'\t' | b'\r' | b'\n') {
        start += 1;
    }
    while end > start && matches!(bytes[end - 1], b' ' | b'\t' | b'\r' | b'\n') {
        end -= 1;
    }

    &bytes[start..end]
}

/// Given the byte index of a `fn <name>` signature, walks forward counting
/// brace depth to find the end of the balanced function body, and returns
/// the extracted `&str` slice for `file_bytes[start_idx..end_idx]`.
///
/// Category: internal invariant. By the time this function runs, the
/// signature has already been located inside a byte slice that was
/// validated as UTF-8 by the caller; a source file with genuinely
/// unbalanced braces (or an underflow, i.e. a stray `}` before any `{`)
/// indicates either malformed/adversarial input or corrupted memory, so
/// both conditions use the full debug_assert + terse-return pattern.
fn extract_balanced_function_body(file_bytes: &[u8], start_idx: usize) -> Result<&str> {
    let mut brace_depth: u32 = 0;
    let mut found_first_brace = false;
    let mut extraction_end_index: Option<usize> = None;

    let mut k = start_idx;
    while k < file_bytes.len() {
        let current_byte = file_bytes[k];

        if current_byte == b'{' {
            found_first_brace = true;
            brace_depth = match brace_depth.checked_add(1) {
                Some(v) => v,
                None => {
                    // Internal invariant: cannot happen with a
                    // MAX_SOURCE_FILE_BYTES-bounded file (nesting depth is
                    // bounded by file length), guarded here defensively.
                    #[cfg(all(debug_assertions, not(test)))]
                    debug_assert!(
                        false,
                        "FNFINDER-300: extract_balanced_function_body: brace depth overflow"
                    );
                    return Err(FnFinderError::BraceStackUnderflow);
                }
            };
        } else if current_byte == b'}' {
            if brace_depth == 0 {
                // Internal invariant: a closing brace with no corresponding
                // open brace should not be reachable once found_first_brace
                // gates entry into counting; defensively checked-return.
                #[cfg(all(debug_assertions, not(test)))]
                debug_assert!(
                    false,
                    "FNFINDER-300: extract_balanced_function_body: brace underflow at byte {}",
                    k
                );

                #[cfg(debug_assertions)]
                eprintln!(
                    "FNFINDER-300: extract_balanced_function_body: brace underflow at byte {}",
                    k
                );

                return Err(FnFinderError::BraceStackUnderflow);
            }
            brace_depth -= 1;

            if found_first_brace && brace_depth == 0 {
                extraction_end_index = Some(k + 1);
                break;
            }
        }

        k += 1;
    }

    let end_idx = match extraction_end_index {
        Some(idx) => idx,
        None => {
            #[cfg(all(debug_assertions, not(test)))]
            debug_assert!(
                false,
                "FNFINDER-301: extract_balanced_function_body: never reached balanced close"
            );

            #[cfg(debug_assertions)]
            eprintln!("FNFINDER-301: extract_balanced_function_body: never reached balanced close");

            return Err(FnFinderError::BraceStackUnterminated);
        }
    };

    match std::str::from_utf8(&file_bytes[start_idx..end_idx]) {
        Ok(slice) => Ok(slice),
        Err(_detail) => {
            #[cfg(debug_assertions)]
            eprintln!(
                "FNFINDER-201: extract_balanced_function_body utf8: {}",
                _detail
            );
            Err(FnFinderError::FileReadFailed)
        }
    }
}

/// Writes a string slice to the terminal one byte at a time, without heap
/// allocation, then flushes.
///
/// Category: input validation / expected external condition. A downstream
/// terminal, pipe, or redirect target can close or fail at any time; this
/// is not a bug in this process, so no `debug_assert!(false, ...)` is used.
pub fn write_to_terminal_char_by_char(text: &str) -> Result<()> {
    let stdout = io::stdout();
    let mut handle = stdout.lock();

    for byte in text.bytes() {
        let single_byte_slice = [byte];
        match handle.write_all(&single_byte_slice) {
            Ok(()) => {}
            Err(_detail) => {
                #[cfg(debug_assertions)]
                eprintln!("FNFINDER-302: write_to_terminal_char_by_char: {}", _detail);
                return Err(FnFinderError::TerminalWriteFailed);
            }
        }
    }

    match handle.flush() {
        Ok(()) => Ok(()),
        Err(_detail) => {
            #[cfg(debug_assertions)]
            eprintln!(
                "FNFINDER-302: write_to_terminal_char_by_char flush: {}",
                _detail
            );
            Err(FnFinderError::TerminalWriteFailed)
        }
    }
}

/// Locates the named function definition inside `rust_file_path` and writes
/// its full source text, character by character, to the terminal.
///
/// This function only orchestrates calls to the smaller, independently
/// tested helper functions above; it performs no direct parsing logic
/// itself, keeping its own cyclomatic complexity low.
pub fn finder_rs_for_terminal(rust_file_path: &Path, function_name_to_find: &str) -> Result<()> {
    match validate_finder_inputs(rust_file_path, function_name_to_find) {
        Ok(()) => {}
        Err(err) => return Err(err),
    }

    let mut stack_buffer = [0u8; MAX_SOURCE_FILE_BYTES];
    let bytes_read = match read_file_into_stack_buffer(rust_file_path, &mut stack_buffer) {
        Ok(n) => n,
        Err(err) => return Err(err),
    };

    let file_bytes = &stack_buffer[..bytes_read];

    let start_idx = match locate_function_signature_start(file_bytes, function_name_to_find) {
        Ok(Some(idx)) => idx,
        Ok(None) => {
            #[cfg(debug_assertions)]
            eprintln!("FNFINDER-105: finder_rs_for_terminal: signature not found");
            return Err(FnFinderError::FunctionSignatureNotFound);
        }
        Err(err) => return Err(err),
    };

    let extraction_start_idx = match find_doc_comment_block_start(file_bytes, start_idx) {
        Ok(idx) => idx,
        Err(err) => return Err(err),
    };

    let extracted_function_slice =
        match extract_balanced_function_body(file_bytes, extraction_start_idx) {
            Ok(slice) => slice,
            Err(err) => return Err(err),
        };

    match write_to_terminal_char_by_char(extracted_function_slice) {
        Ok(()) => {}
        Err(err) => return Err(err),
    }

    match write_to_terminal_char_by_char("\n") {
        Ok(()) => {}
        Err(err) => return Err(err),
    }

    Ok(())
}

/// Binary execution entrypoint.
///
/// Note: `std::env::args()` and command-line `String` values are heap
/// allocated by the standard library at process start, independent of this
/// framework's error-path no-heap policy; the policy governs case/error
/// handling paths inside project functions, not unavoidable OS/runtime
/// argument plumbing.
fn main() {
    let mut args = std::env::args();
    let _exec_name = args.next();

    let file_path_str = match args.next() {
        Some(path) => path,
        None => {
            #[cfg(debug_assertions)]
            eprintln!("Usage: definder <file_path.rs> <function_name>");
            std::process::exit(1);
        }
    };

    let target_function = match args.next() {
        Some(func) => func,
        None => {
            #[cfg(debug_assertions)]
            eprintln!("Usage: definder <file_path.rs> <function_name>");
            std::process::exit(1);
        }
    };

    let path = Path::new(&file_path_str);
    match finder_rs_for_terminal(path, &target_function) {
        Ok(()) => {}
        Err(err) => {
            #[cfg(debug_assertions)]
            eprintln!("Execution failed with error code: {}", err.code());
            std::process::exit(err.code() as i32);
        }
    }
}

#[cfg(test)]
mod finder_rs_for_terminal_tests {
    use super::*;

    // #[test]
    // fn u16td_converts_boundary_values_with_no_code() {
    //     let mut buf = [0u8; 5];
    //     assert_eq!(u16_to_decimal(101, &mut buf), "101");
    //     assert_eq!(u16_to_decimal(0, &mut buf), "0");
    //     assert_eq!(u16_to_decimal(65535, &mut buf), "65535");
    // }

    #[test]
    fn fferr_code_and_retry_classification_are_consistent() {
        let err = FnFinderError::PathIsEmpty;
        assert_eq!(err.code(), 101);
        assert!(!err.is_retryable());

        let io_err = FnFinderError::FileOpenFailed;
        assert_eq!(io_err.code(), 200);
        assert!(io_err.is_retryable());
    }

    #[test]
    fn vfi_rejects_empty_function_name_with_code_102() {
        let dummy_path = Path::new("nonexistent.rs");
        let result = validate_finder_inputs(dummy_path, "");
        assert_eq!(result, Err(FnFinderError::FunctionNameIsEmpty));
        assert_eq!(result.unwrap_err().code(), 102);
    }

    #[test]
    fn vfi_rejects_empty_path_with_code_101() {
        let empty_path = Path::new("");
        let result = validate_finder_inputs(empty_path, "some_fn");
        assert_eq!(result, Err(FnFinderError::PathIsEmpty));
    }

    #[test]
    fn vfi_rejects_missing_file_with_code_103() {
        let missing_path = Path::new("/definitely/does/not/exist_ghost_file.rs");
        let result = validate_finder_inputs(missing_path, "some_fn");
        assert_eq!(result, Err(FnFinderError::TargetFileNotFound));
    }

    #[test]
    fn lfss_returns_none_when_signature_absent() {
        let content = b"struct Thing;\nfn other() {}\n";
        let result = locate_function_signature_start(content, "target_fn");
        assert_eq!(result, Ok(None));
    }

    #[test]
    fn lfss_finds_signature_start_index() {
        let content = b"struct Thing;\nfn target_fn() {}\n";
        let result = locate_function_signature_start(content, "target_fn");
        assert_eq!(result, Ok(Some(14)));
    }

    #[test]
    fn ebfb_extracts_balanced_single_brace_body() {
        let content = b"fn target_fn() { let a = 1; }";
        let extracted = extract_balanced_function_body(content, 0).unwrap();
        assert_eq!(extracted, "fn target_fn() { let a = 1; }");
    }

    #[test]
    fn ebfb_extracts_balanced_nested_brace_body() {
        let content = b"fn target_fn() { if true { let a = 1; } }";
        let extracted = extract_balanced_function_body(content, 0).unwrap();
        assert_eq!(extracted, content_as_str(content));
    }

    fn content_as_str(content: &[u8]) -> &str {
        std::str::from_utf8(content).unwrap()
    }

    #[test]
    fn ebfb_catches_unterminated_body_with_code_301() {
        let content = b"fn target_fn() { let a = 1;";
        let result = extract_balanced_function_body(content, 0);
        assert_eq!(result, Err(FnFinderError::BraceStackUnterminated));
        assert_eq!(result.unwrap_err().code(), 301);
    }

    #[test]
    fn ebfb_catches_stray_closing_brace_with_code_300() {
        // Simulates a corrupted/adversarial slice with a closing brace
        // before any opening brace is encountered.
        let content = b"fn target_fn() } let a = 1; {";
        let result = extract_balanced_function_body(content, 0);
        assert_eq!(result, Err(FnFinderError::BraceStackUnderflow));
        assert_eq!(result.unwrap_err().code(), 300);
    }

    #[test]
    fn end_to_end_rejects_missing_function_name_with_code_105() {
        // Uses this very source file as the target, searching for a
        // function name guaranteed not to exist.
        let this_file = Path::new(file!());
        let result = finder_rs_for_terminal(this_file, "definitely_not_a_real_function_xyz");
        assert_eq!(result, Err(FnFinderError::FunctionSignatureNotFound));
    }

    #[test]
    fn fdcbs_includes_contiguous_doc_comment_lines() {
        let content = b"/// Line one.\n/// Line two.\nfn target_fn() {}\n";
        let sig_start = locate_function_signature_start(content, "target_fn")
            .unwrap()
            .unwrap();
        let doc_start = find_doc_comment_block_start(content, sig_start).unwrap();
        let extracted = extract_balanced_function_body(content, doc_start).unwrap();
        assert_eq!(extracted, "/// Line one.\n/// Line two.\nfn target_fn() {}");
    }

    #[test]
    fn fdcbs_stops_at_blank_line_separator() {
        let content = b"/// Unrelated doc.\n\nfn target_fn() {}\n";
        let sig_start = locate_function_signature_start(content, "target_fn")
            .unwrap()
            .unwrap();
        let doc_start = find_doc_comment_block_start(content, sig_start).unwrap();
        assert_eq!(doc_start, sig_start);
    }

    #[test]
    fn fdcbs_returns_signature_start_when_no_doc_comment_present() {
        let content = b"struct Thing;\nfn target_fn() {}\n";
        let sig_start = locate_function_signature_start(content, "target_fn")
            .unwrap()
            .unwrap();
        let doc_start = find_doc_comment_block_start(content, sig_start).unwrap();
        assert_eq!(doc_start, sig_start);
    }

    #[test]
    fn fdcbs_includes_three_contiguous_doc_comment_lines() {
        // Regression test for the original bug: only the immediately
        // adjacent doc-comment line was included, not the full contiguous
        // block, because the backward-walking cursor was not consistently
        // maintained across iterations.
        let content = b"/// Line one.\n/// Line two.\n/// Line three.\nfn target_fn() {}\n";
        let sig_start = locate_function_signature_start(content, "target_fn")
            .unwrap()
            .unwrap();
        let doc_start = find_doc_comment_block_start(content, sig_start).unwrap();
        let extracted = extract_balanced_function_body(content, doc_start).unwrap();
        assert_eq!(
            extracted,
            "/// Line one.\n/// Line two.\n/// Line three.\nfn target_fn() {}"
        );
    }
}
