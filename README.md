# function_finder (for Rust)

A .py tool (function_finder.py) with a cli Q&A interface for use \
with Rust files (.rs) and code directories (e.g. /src/)
to:

##### A. Extract a function or nest of functions from a directory or file
- primary function
- up to N depth of the same for functions called by that function
- in up to N depth sub-directories of modules in the code directory

##### B. Save those functions to individual files and optionally to one flat file

The goal of this tool is to be useful for working with a large code project \
where a routine but non-trivial task is extracting the code-scope \
around a specific function.

#### Simple Tool:
This is a simple tool that does not cover all possible edge cases. \
An error-log file is produced to provide possible exception data.


### Function 1: `function_finder()`

#### Signature
```python
function_finder(
    rust_code_dir_path: str,
    function_name_to_find: str,
    function_depth: int = 0,
    file_depth: int = 2,
    only_search_this_file_path: str | None = None,
    output_dir: str | None = None,
) -> list[str]
```

## Function 2: `flatten_finder(dir_path)`

#### Signature
```python
flatten_finder(dir_path: str) -> str
```
