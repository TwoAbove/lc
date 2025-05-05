#!/usr/bin/env -S uv --quiet run --script
# /// script
# dependencies = [
#   "pyperclip",
#   "colorama",
#   "tiktoken",
#   "pathspec",
#   "openai"
# ]
# ///

import os
import json
import openai
import sys
import re
import pyperclip
import argparse
import tiktoken
from sys import platform
from pathlib import Path
from typing import Set, List, Dict, Any, Optional, Tuple
from colorama import init, Fore, Style
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern

init(autoreset=True)

INSTRUCTIONS_TEXT = """
This document contains a representation of one or more codebases.
Each codebase is enclosed in <codebase> tags with a 'path' attribute.
Files are represented by <file> tags with the 'path' attribute.
File contents are stored within the <file> tags.
For directory-only mode, <directory> tags are used instead of <file> tags.
"""

BINARY_EXTENSIONS = {
    ".wasm",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    ".webp",
    ".tiff",
    '.tff',
    '.woff',
    '.woff2',
    ".tif",
    ".psd",
    ".raw",
    ".heif",
    ".indd",
    ".ai",
    ".eps",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".mp3",
    ".flac",
    ".wav",
    ".aac",
    ".wma",
    ".ogg",
    ".mp4",
    '.m4a',
    ".mkv",
    ".webm",
    ".avi",
    ".mov",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".flv",
    ".3gp",
    ".zip",
    ".rar",
    ".7z",
    ".gz",
    ".tar",
    ".tgz",
    ".bz2",
    ".xz",
    ".lz",
    ".lz4",
    ".lzo",
    ".zst",
    ".zstd",
    ".z",
    ".tar.gz",
    ".tar.xz",
    ".tar.bz2",
    ".tar.lz",
    ".tar.lz4",
    ".tar.lzo",
    ".tar.zst",
    ".tar.zstd",
    ".tar.z",
}


class TokenCounter:
    def __init__(self, model_name: str = "o200k_base"):
        self.encoder = tiktoken.get_encoding(model_name)

    def count_tokens(self, text: str) -> int:
        try:
            return len(self.encoder.encode(text))
        except Exception:
            return -1


class LCDocument:
    def __init__(self):
        self.codebases = []
        self.instructions = INSTRUCTIONS_TEXT.strip()

    def add_or_update_codebase(self, codebase):
        for i, existing in enumerate(self.codebases):
            if existing.path == codebase.path:
                self.codebases[i] = codebase
                return
        self.codebases.append(codebase)

    def to_string(self) -> str:
        lines = []
        lines.append("<lc>")
        lines.append("<instructions>")
        lines.append(self.instructions)
        lines.append("</instructions>")

        for codebase in self.codebases:
            lines.append(codebase.to_string())

        lines.append("</lc>")
        return "\n".join(lines)

    @classmethod
    def from_string(cls, content: str) -> Optional["LCDocument"]:
        if not content or "<lc>" not in content:
            return None

        doc = cls()

        # Extract instructions if present
        instructions_match = re.search(
            r"<instructions>(.*?)</instructions>", content, re.DOTALL
        )
        if instructions_match:
            doc.instructions = instructions_match.group(1).strip()

        # Extract codebases
        codebase_pattern = r'<codebase\s+path="([^"]+)">(.*?)</codebase>'
        for match in re.finditer(codebase_pattern, content, re.DOTALL):
            path = match.group(1)
            codebase_content = match.group(2)
            codebase = Codebase.from_string(path, codebase_content)
            doc.codebases.append(codebase)

        return doc


class Codebase:
    def __init__(self, path: str):
        self.path = path
        self.entries = []  # List[FileEntry or DirectoryEntry]

    def add_entry(self, entry):
        self.entries.append(entry)

    def to_string(self) -> str:
        lines = []
        lines.append(f'<codebase path="{self.path}">')
        for entry in self.entries:
            lines.append(entry.to_string())
        lines.append("</codebase>")
        return "\n".join(lines)

    @classmethod
    def from_string(cls, path: str, content: str) -> "Codebase":
        codebase = cls(path)

        # Extract files
        file_pattern = r'<file\s+path="([^"]+)"\s+tokens="(\d+)">(.*?)</file>'
        for match in re.finditer(file_pattern, content, re.DOTALL):
            file_path = match.group(1)
            tokens = int(match.group(2))
            file_content = match.group(3)
            entry = FileEntry(file_path, file_content, tokens)
            codebase.add_entry(entry)

        # Extract directories
        dir_pattern = r'<directory\s+path="([^"]+)"\s+tokens="(\d+)".*?</directory>'
        for match in re.finditer(dir_pattern, content):
            dir_path = match.group(1)
            tokens = int(match.group(2))
            entry = DirectoryEntry(dir_path, tokens)
            codebase.add_entry(entry)

        return codebase


class FileEntry:
    def __init__(self, path: str, content: str, tokens: int):
        self.path = path
        self.content = content
        self.tokens = tokens
        self.lines = len(content.splitlines()) if content else 0

    def to_string(self) -> str:
        return f'<file path="{self.path}" tokens="{self.tokens}">{self.content}</file>'


class DirectoryEntry:
    def __init__(self, path: str, tokens: int = 0):
        self.path = path
        self.tokens = tokens

    def to_string(self) -> str:
        return f'<directory path="{self.path}" tokens="{self.tokens}"></directory>'


class CodebaseTraverser:
    def __init__(
        self,
        directory: Path,
        ignore_patterns: Set[str],
        directory_only: bool,
        token_limit: Optional[int] = None,
    ):
        self.directory = directory
        self.pathspec = PathSpec.from_lines(GitWildMatchPattern, ignore_patterns)
        self.directory_only = directory_only
        self.token_counter = TokenCounter()
        self.token_limit = token_limit
        self.large_files: List[Tuple[str, int]] = []
        self.has_token_errors = False
        self.files_with_token_errors: List[str] = []

    def traverse(self) -> List[Dict[str, Any]]:
        codebase = []
        for root, dirs, files in os.walk(str(self.directory), followlinks=True):
            rel_root = Path(root).relative_to(self.directory)

            for dir_name in dirs[:]:
                dir_path = rel_root / dir_name
                if self.pathspec.match_file(str(dir_path)):
                    dirs.remove(dir_name)
                elif self.directory_only:
                    codebase.append(
                        {"path": str(dir_path), "content": "", "lines": 0, "tokens": 0}
                    )

            if not self.directory_only:
                for file_name in files:
                    file_path = rel_root / file_name
                    if not self.pathspec.match_file(str(file_path)):
                        full_path = Path(root) / file_name
                        file_info = {
                            "path": str(file_path),
                            "content": "",
                            "lines": 0,
                            "tokens": 0,
                        }
                        self._process_file(full_path, file_info)
                        codebase.append(file_info)

        return codebase

    def _is_binary_file(self, file_path: str) -> bool:
        return file_path.lower().endswith(tuple(BINARY_EXTENSIONS))

    def _process_file(self, item: Path, file_info: Dict[str, Any]):
        if self._is_binary_file(str(item)):
            file_info["content"] = "[Binary file]"
            file_info["tokens"] = 2
        else:
            try:
                with open(item, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    file_info["content"] = content
                    file_info["lines"] = len(content.splitlines())

                    token_count = self.token_counter.count_tokens(content)
                    if token_count == -1:
                        self.has_token_errors = True
                        self.files_with_token_errors.append(str(item))
                        file_info["tokens"] = 0
                    else:
                        file_info["tokens"] = token_count
                        if self.token_limit and token_count > self.token_limit:
                            self.large_files.append((str(item), token_count))
            except Exception as e:
                print(f"Error reading file {item}: {e}", file=sys.stderr)
                file_info["content"] = f"Error reading file: {e}"
                token_count = self.token_counter.count_tokens(file_info["content"])
                if token_count == -1:
                    self.has_token_errors = True
                    self.files_with_token_errors.append(str(item))
                    file_info["tokens"] = 0
                else:
                    file_info["tokens"] = token_count


class SimpleGenerator:
    @staticmethod
    def generate(
        codebase_entries: List[Dict[str, Any]], pwd: str, directory_only: bool
    ) -> str:
        codebase = Codebase(pwd)

        for entry in codebase_entries:
            if directory_only:
                codebase.add_entry(DirectoryEntry(entry["path"], entry["tokens"]))
            else:
                codebase.add_entry(
                    FileEntry(entry["path"], entry["content"], entry["tokens"])
                )

        doc = LCDocument()
        doc.add_or_update_codebase(codebase)
        return doc.to_string()


def find_git_root(start_path: Path) -> Optional[Path]:
    current_path = start_path.resolve()
    while current_path != current_path.parent:
        if (current_path / ".git").is_dir():
            return current_path
        current_path = current_path.parent
    return None


def parse_ignore_file(file_path: Path) -> Set[str]:
    ignore_patterns = set()
    if file_path.exists():
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if line.startswith("/"):
                        line = line[1:]
                    ignore_patterns.add(line)
    return ignore_patterns


def get_ignore_patterns(base_directory: Path, git_root: Optional[Path]) -> Set[str]:
    ignore_patterns = set()
    if git_root:
        ignore_patterns.update(parse_ignore_file(git_root / ".gitignore"))
        ignore_patterns.update(parse_ignore_file(git_root / ".repoignore"))
    else:
        ignore_patterns.update(parse_ignore_file(base_directory / ".gitignore"))
        ignore_patterns.update(parse_ignore_file(base_directory / ".repoignore"))

    # Also add home dir repoignore
    ignore_patterns.update(parse_ignore_file(Path.home() / ".repoignore"))

    ignore_patterns.update({".git", ".repo", "package-lock.json", "yarn.lock"})
    return ignore_patterns


def get_stats_from_content(content: str, directory_only: bool) -> Dict[str, Any]:
    doc = LCDocument.from_string(content)
    if not doc:
        return {"files": 0, "tokens": 0, "lines": 0, "has_token_errors": False}

    total_files = 0
    total_tokens = 0
    total_lines = 0
    has_token_errors = False

    for codebase in doc.codebases:
        for entry in codebase.entries:
            total_files += 1
            total_tokens += entry.tokens
            if isinstance(entry, FileEntry):
                total_lines += entry.lines

    return {
        "files": total_files,
        "tokens": total_tokens,
        "lines": total_lines if not directory_only else 0,
        "has_token_errors": has_token_errors
    }


class Config:
    DEFAULT_MODEL = "openai/gpt-4.1"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOKEN_LIMIT = 10000

    @classmethod
    def get_api_key(cls) -> str:
        return os.getenv("OPENROUTER_API_KEY", "")

    @classmethod
    def get_api_base_url(cls) -> str:
        return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def get_openai_client() -> openai.OpenAI:
    """Return configured OpenAI client."""
    return openai.OpenAI(
        base_url=Config.get_api_base_url(),
        api_key=Config.get_api_key()
    )


def generate_command(prompt: str) -> str:
    """
    Generate a shell command using OpenAI's API based on the given prompt.
    """
    system_prompt = f"""You are an expert in writing terminal commands. Your task is to convert natural language requests into shell commands.
Rules:
1. Only return the command itself, no explanations or comments
2. Use common Unix/Linux commands - the user us running {platform}
3. Ensure the command is safe and won't cause damage. If the command could be dangerous, return 'echo "Unsafe command"' unless the user explicitly asks for it
4. If unsure, return 'echo "Unable to generate safe command for this request"'
5. Don't use sudo unless explicitly requested
6. Prefer command chaining over scripts
7. Use relative paths when possible

Example inputs/outputs:
Input: "show all PDF files recursively"
Output: find . -name "*.pdf"

Input: "what's using port 8080"
Output: lsof -i :8080

Input: "zip all jpg files"
Output: find . -name "*.jpg" -exec zip images.zip {"{}"} +
"""

    try:
        client = get_openai_client()

        if not Config.get_api_key():
            return 'echo "Error: OPENROUTER_API_KEY environment variable not set"'

        response = client.chat.completions.create(
            model=Config.DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=Config.DEFAULT_TEMPERATURE
        )

        command = response.choices[0].message.content.strip()
        return command
    except openai.APIError as e:
        return f'echo "API Error: {str(e)}"'
    except openai.APIConnectionError as e:
        return f'echo "Connection Error: Check your network connection"'
    except openai.RateLimitError:
        return f'echo "Rate limit exceeded: Please try again later"'
    except Exception as e:
        return f'echo "Error generating command: {str(e)}"'


def print_stats(content: str, directory_only: bool, large_files: List[Tuple[str, int]],
                has_token_errors: bool = False, files_with_token_errors: List[str] = None):
    stats = get_stats_from_content(content, directory_only)
    files_with_token_errors = files_with_token_errors or []
    has_token_errors = has_token_errors or stats.get("has_token_errors", False)

    # Display summary stats based on directory_only mode
    if directory_only:
        print(f"d: {Fore.GREEN}{stats['files']}{Style.RESET_ALL}")
    else:
        tokens_display = f"{stats['tokens']} + ???" if has_token_errors else f"{stats['tokens']}"
        print(
            f"f: {Fore.GREEN}{stats['files']}{Style.RESET_ALL} "
            f"l: {Fore.YELLOW}{stats['lines']}{Style.RESET_ALL} "
            f"t: {Fore.MAGENTA}{tokens_display}{Style.RESET_ALL}"
        )

    # Display warnings for large files if any
    if large_files:
        print(f"\n{Fore.RED}Files exceeding token limit:{Style.RESET_ALL}")
        for file_path, token_count in large_files:
            print(
                f"{Fore.YELLOW}{file_path}{Style.RESET_ALL}: {Fore.MAGENTA}{token_count}{Style.RESET_ALL} tokens"
            )

    # Display files with token counting errors if any
    if files_with_token_errors:
        print(f"\n{Fore.RED}Files with failed token counts:{Style.RESET_ALL}")
        for file_path in files_with_token_errors:
            print(f"{Fore.YELLOW}{file_path}{Style.RESET_ALL}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate structured output from directory for multiple codebases or handle command execution."
    )

    # Create mutually exclusive group for main operations
    operation_group = parser.add_mutually_exclusive_group()

    # Regular codebase operation flags
    operation_group.add_argument(
        "-c", "--copy",
        action="store_true",
        help="Copy codebase representation to clipboard (default behavior)",
    )

    # Command generation/execution flags
    operation_group.add_argument(
        "-e", "--execute",
        type=str,
        help="Generate a command from the prompt and copy to clipboard",
    )
    operation_group.add_argument(
        "-ee", "--execute-direct",
        type=str,
        help="Generate and directly execute the command",
    )

    # Other arguments
    parser.add_argument(
        "subfolder",
        nargs="?",
        default=".",
        help="Subfolder to process (default: current directory)",
    )
    parser.add_argument(
        "-d",
        "--directory-only",
        action="store_true",
        help="Output only directory structure without file contents",
    )
    parser.add_argument(
        "-t",
        "--token-limit",
        default=Config.DEFAULT_TOKEN_LIMIT,
        type=int,
        help="Token limit per file (warns if exceeded)",
    )
    args = parser.parse_args()

    if args.execute or args.execute_direct:
        # Handle command generation/execution
        prompt = args.execute or args.execute_direct
        command = generate_command(prompt)

        if args.execute_direct:
            print(f"{Fore.YELLOW}Executing command:{Style.RESET_ALL} {command}")
            try:
                os.system(command)
            except Exception as e:
                print(f"{Fore.RED}Error executing command:{Style.RESET_ALL} {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"{Fore.GREEN}Generated command copied to clipboard:{Style.RESET_ALL} {command}")
            pyperclip.copy(command)
    else:
        # Handle regular codebase operation
        base_directory = Path.cwd()
        directory_path = (base_directory / args.subfolder).resolve()

        if not directory_path.exists() or not directory_path.is_dir():
            print(f"Error: {directory_path} is not a valid directory.", file=sys.stderr)
            sys.exit(1)

        git_root = find_git_root(directory_path)
        ignore_patterns = get_ignore_patterns(base_directory, git_root)

        traverser = CodebaseTraverser(
            directory_path, ignore_patterns, args.directory_only, args.token_limit
        )
        codebase_entries = traverser.traverse()

        # Generate new codebase content
        new_content = SimpleGenerator.generate(
            codebase_entries, str(directory_path), args.directory_only
        )

        # Get existing clipboard content and parse it if it's an LC document
        clipboard_content = pyperclip.paste()
        existing_doc = LCDocument.from_string(clipboard_content)

        if existing_doc:
            # Parse the new content as a document
            new_doc = LCDocument.from_string(new_content)
            if new_doc and new_doc.codebases:
                # Update or add the new codebase
                existing_doc.add_or_update_codebase(new_doc.codebases[0])
                final_content = existing_doc.to_string()
            else:
                final_content = new_content
        else:
            final_content = new_content

        # Print statistics
        print_stats(
            final_content,
            args.directory_only,
            traverser.large_files,
            traverser.has_token_errors,
            traverser.files_with_token_errors
        )

        # Update clipboard
        pyperclip.copy(final_content)


if __name__ == "__main__":
    main()
