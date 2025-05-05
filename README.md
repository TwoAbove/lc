# lc

A CLI tool to accelerate LLM usage by copying codebase snapshots (`lc`) or generating commands from prompts (`lc -e`). Additional QoL features planned as I find the uses for them.

lc creates a structured XML-like document of codebases for LLMs, respecting .gitignore and supporting multiple codebases.

## Features

- 🌳 Use with multiple codebases easily
- 🚫 Respects `.gitignore` and `.repoignore` patterns
- 📊 Token counting and warnings to keep within token limits
- 📋 Automatic clipboard integration
- 🛡️ Binary file detection and handling
- 💾 Preserves existing codebase contexts in clipboard
- 🖥️ Command generation with `lc -e`

## Prerequisites

- Python 3.7+
- [uv](https://github.com/astral-sh/uv) package manager
- Required Python packages (automatically installed by uv):
  - pyperclip
  - lxml
  - colorama
  - tiktoken
  - pathspec
  - openai

## Installation

1. Install [uv](https://github.com/astral-sh/uv) if you haven't already.

2. Clone this repository:
```bash
git clone https://github.com/TwoAbove/lc
cd lc
```

3. Add the following function to your shell configuration file (`.bashrc`, `.zshrc`, etc.):
```bash
lc() {
  local script_path="<PATH>/<TO>/lc.py"
  if [ -x "$script_path" ]; then
    "$script_path" "$@"
  else
    echo "Error: lc.py not found or not executable at $script_path"
    return 1
  fi
}
```

4. Reload your shell configuration:
```bash
source ~/.bashrc  # or ~/.zshrc
```

5. Use it in your codebase!
```bash
lc
```

## Usage

### Basic Usage

```bash
# Process current directory
lc

# Process specific subdirectory
lc path/to/directory

# Directory-only mode (no file contents)
lc -d

# Set custom token limit warning
lc -t 5000

# Generate command from prompt
lc -e "list all python files"
```

### Output Format

The tool outputs colored statistics in the terminal:
- `f`: Number of files
- `l`: Total lines of code
- `t`: Total tokens

Files exceeding the token limit are listed separately in red.

### Working with Multiple Codebases

The tool preserves existing context in your clipboard, allowing you to build up context from multiple codebases:

1. Run `lc` in first project directory
2. Change to another project directory
3. Run `lc` again
4. The clipboard now contains both codebases in a single XML document

### Best Practices

1. **Use `.repoignore`**
   - Create a `.repoignore` file in your repository root
   - Add patterns for files you don't want to include in the context
   - Similar syntax to `.gitignore`
   - Useful for excluding large generated files, logs, etc.
   - Add a global `.repoignore` in your home directory to ignore common patterns

2. **Token Management**
   - Default token limit is 10,000 per file - mostly a sanity check
   - Adjust using `-t` flag based on your LLM's context window
   - Monitor the token count in the output

3. **Directory Structure**
   - Use directory-only mode (`-d`) for initial exploration of large codebases
   - Helps manage token usage while providing structural context

### Gotchas and Notes

1. **Binary Files**
   - Automatically detected and marked as `[Binary file]`
   - Common binary extensions are pre-configured
   - If some common binary extensions are missing, please send a PR!
   - Helps prevent clipboard overuse and saves tokens

2. **Clipboard Behavior**
   - Always reads existing clipboard content
   - Updates existing codebase context if path matches
   - Adds new codebase context if path is different

3. **File Size**
   - Large files are reported but still included
   - Consider using `.repoignore` for consistently large files or files that are not useful for LLMs

4. **Git Integration**
   - Automatically finds Git root directory
   - Respects `.gitignore` patterns from repository root
   - Works in non-Git directories too

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)
