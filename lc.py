#!/usr/bin/env -S uv run

# /// script
# dependencies = [
#   "pyperclip",
#   "lxml",
#   "colorama",
#   "tiktoken",
#   "pathspec"
# ]
# ///

import os
import sys
import pyperclip
import argparse
import tiktoken
from pathlib import Path
from typing import Set, List, Dict, Any, Optional, Tuple
from lxml import etree
from colorama import init, Fore, Style
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern

# Initialize colorama
init(autoreset=True)

INSTRUCTIONS_TEXT = """
This XML document contains a representation of one or more codebases.
Each codebase is enclosed in a <codebase> tag with a 'path' attribute.
Files are represented by <file> tags with the 'path' attribute.
File contents are stored as CDATA within the <file> tags.
For directory-only mode, <directory> tags are used instead of <file> tags.
"""

BINARY_EXTENSIONS = {
    "wasm",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    ".webp",
    ".tiff",
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
    def __init__(self, model_name: str = "cl100k_base"):
        self.encoder = tiktoken.get_encoding(model_name)

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))


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

    def traverse(self) -> List[Dict[str, Any]]:
        codebase = []
        for root, dirs, files in os.walk(str(self.directory), followlinks=True):
            rel_root = Path(root).relative_to(self.directory)

            # Process directories
            for dir_name in dirs[:]:
                dir_path = rel_root / dir_name
                if self.pathspec.match_file(str(dir_path)):
                    dirs.remove(dir_name)
                elif self.directory_only:
                    codebase.append(
                        {"path": str(dir_path), "content": "", "lines": 0, "tokens": 0}
                    )

            # Process files
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

    def _should_ignore(self, file_path: str) -> bool:
        return self.pathspec.match_file(file_path)

    def _is_binary_file(self, file_path: str) -> bool:
        return file_path.lower().endswith(tuple(BINARY_EXTENSIONS))

    def _process_file(self, item: Path, file_info: Dict[str, Any]):
        if self._is_binary_file(str(item)):
            file_info["content"] = "[Binary file]"
            file_info["tokens"] = 2  # Count "[Binary file]" as 2 tokens
        else:
            try:
                with open(item, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    try:
                        etree.CDATA(content)
                        file_info["content"] = content
                        file_info["lines"] = len(content.splitlines())
                        file_info["tokens"] = self.token_counter.count_tokens(content)

                        if self.token_limit and file_info["tokens"] > self.token_limit:
                            self.large_files.append((str(item), file_info["tokens"]))

                    except ValueError as e:
                        print(
                            f"{Fore.RED}XML incompatible characters in file: {Fore.YELLOW}{item}{Style.RESET_ALL}"
                        )
                        file_info["content"] = "[Binary file]"
                        file_info["lines"] = 0
                        file_info["tokens"] = 2
            except Exception as e:
                print(f"Error reading file {item}: {e}", file=sys.stderr)
                file_info["content"] = f"Error reading file: {e}"
                file_info["tokens"] = self.token_counter.count_tokens(
                    file_info["content"]
                )


class XMLGenerator:
    @staticmethod
    def generate(
        codebase: List[Dict[str, Any]], pwd: str, directory_only: bool
    ) -> etree.Element:
        try:
            root = etree.Element("lc")
            XMLGenerator._add_instructions(root)
            codebase_elem = etree.SubElement(root, "codebase", path=pwd)

            for file_info in codebase:
                XMLGenerator._add_file_or_directory(
                    codebase_elem, file_info, directory_only
                )

            return root
        except Exception as e:
            print(f"Error generating XML: {e}", file=sys.stderr)
            sys.exit(1)

    @staticmethod
    def _add_instructions(root: etree.Element):
        instructions = etree.SubElement(root, "instructions")
        instructions.text = etree.CDATA(INSTRUCTIONS_TEXT)

    @staticmethod
    def _add_file_or_directory(
        parent: etree.Element, file_info: Dict[str, Any], directory_only: bool
    ):
        elem_type = "directory" if directory_only else "file"
        file_elem = etree.SubElement(
            parent, elem_type, path=file_info["path"], tokens=str(file_info["tokens"])
        )
        if not directory_only and file_info["content"]:
            file_elem.text = etree.CDATA(f"\n{file_info['content']}\n")


class XMLManager:
    @staticmethod
    def update_or_add_codebase(
        existing_xml: str, new_codebase: etree.Element
    ) -> etree.Element:
        root = None
        try:
            # Try parsing as bytes first
            root = etree.fromstring(existing_xml.encode("utf-8"))
        except etree.XMLSyntaxError:
            try:
                # If that fails, try parsing as a string without encoding declaration
                root = etree.fromstring(
                    existing_xml.encode("utf-8"), parser=etree.XMLParser(recover=True)
                )
            except etree.XMLSyntaxError:
                # If both fail, create a new root element
                pass

        if root is None or root.find(".//instructions") is None:
            root = etree.Element("lc")
            XMLGenerator._add_instructions(root)

        new_pwd = new_codebase.find("codebase").get("path")
        existing_codebase = root.find(f".//codebase[@path='{new_pwd}']")

        if existing_codebase is not None:
            root.remove(existing_codebase)
        root.append(new_codebase.find("codebase"))

        return root


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
                    # Ensure patterns starting with / are relative to repo root
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

    ignore_patterns.update({".git", ".repo", "package-lock.json", "yarn.lock"})
    return ignore_patterns


def get_codebase_stats(codebase: etree.Element, directory_only: bool) -> Dict[str, Any]:
    files = codebase.findall(".//file" if not directory_only else ".//directory")
    total_tokens = sum(int(file.get("tokens", 0)) for file in files)

    stats = {
        "files" if not directory_only else "directories": len(files),
        "lines": (
            sum(int(file.get("lines", 0)) for file in codebase.findall(".//file"))
            if not directory_only
            else 0
        ),
        "tokens": total_tokens,
    }
    return stats


def print_all_codebase_stats(
    root: etree.Element, directory_only: bool, large_files: List[Tuple[str, int]]
):
    for codebase in root.findall(".//codebase"):
        stats = get_codebase_stats(codebase, directory_only)
        print(f"{Fore.CYAN}{Style.BRIGHT}{codebase.get('path')}{Style.RESET_ALL}")
        if directory_only:
            print(f"d: {Fore.GREEN}{stats['directories']}{Style.RESET_ALL}")
        else:
            print(
                f"f: {Fore.GREEN}{stats['files']}{Style.RESET_ALL} "
                f"l: {Fore.YELLOW}{stats['lines']}{Style.RESET_ALL} "
                f"t: {Fore.MAGENTA}{stats['tokens']}{Style.RESET_ALL}"
            )

    if large_files:
        print(f"\n{Fore.RED}Files exceeding token limit:{Style.RESET_ALL}")
        for file_path, token_count in large_files:
            print(
                f"{Fore.YELLOW}{file_path}{Style.RESET_ALL}: {Fore.MAGENTA}{token_count}{Style.RESET_ALL} tokens"
            )


def generate_final_xml(root: etree.Element) -> str:
    # Generate XML string without declaration
    xml_string = etree.tostring(root, encoding="unicode", pretty_print=True)

    # Manually add XML declaration
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_string}'


def main():
    parser = argparse.ArgumentParser(
        description="Generate XML from directory structure with dense output for multiple codebases."
    )
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
        help="Output only the directory structure without file contents",
    )
    parser.add_argument(
        "-t",
        "--token-limit",
        default=10000,
        type=int,
        help="Token limit per file (warns if exceeded)",
    )
    args = parser.parse_args()

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
    codebase = traverser.traverse()

    new_codebase_xml = XMLGenerator.generate(
        codebase, str(directory_path), args.directory_only
    )

    clipboard_content = pyperclip.paste()
    updated_xml_root = XMLManager.update_or_add_codebase(
        clipboard_content, new_codebase_xml
    )

    print_all_codebase_stats(
        updated_xml_root, args.directory_only, traverser.large_files
    )

    final_xml = generate_final_xml(updated_xml_root)
    pyperclip.copy(final_xml)


if __name__ == "__main__":
    main()
