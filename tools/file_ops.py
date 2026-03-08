"""
File Operations Tool - Read, write, and manage files
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class FileInfo:
    path: str
    exists: bool
    is_file: bool
    is_dir: bool
    size: int = 0
    extension: str = ""


class FileOpsTool:
    """Read, write, and manage files in the project."""

    name = "file_operations"
    description = "Read, write, list, and manage files. Use for viewing code, creating files, or exploring project structure."

    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to base_path."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_path / p

    def read_file(self, path: str, max_lines: Optional[int] = None) -> str:
        """
        Read contents of a file.

        Args:
            path: File path (absolute or relative to base_path)
            max_lines: Maximum lines to read (None = all)

        Returns:
            File contents as string
        """
        file_path = self._resolve_path(path)

        if not file_path.exists():
            return f"Error: File not found: {path}"

        if not file_path.is_file():
            return f"Error: Not a file: {path}"

        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                if max_lines:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            lines.append(f"\n... (truncated at {max_lines} lines)")
                            break
                        lines.append(line)
                    return ''.join(lines)
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def write_file(self, path: str, content: str, create_dirs: bool = True) -> str:
        """
        Write content to a file.

        Args:
            path: File path
            content: Content to write
            create_dirs: Create parent directories if needed

        Returns:
            Success or error message
        """
        file_path = self._resolve_path(path)

        try:
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return f"Successfully wrote {len(content)} characters to {path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    def list_directory(self, path: str = ".", pattern: str = "*", recursive: bool = False) -> list[FileInfo]:
        """
        List files in a directory.

        Args:
            path: Directory path
            pattern: Glob pattern (e.g., "*.php", "*.blade.php")
            recursive: Search recursively

        Returns:
            List of FileInfo objects
        """
        dir_path = self._resolve_path(path)

        if not dir_path.exists():
            return []

        if not dir_path.is_dir():
            return []

        results = []
        try:
            if recursive:
                files = dir_path.rglob(pattern)
            else:
                files = dir_path.glob(pattern)

            for f in files:
                info = FileInfo(
                    path=str(f.relative_to(self.base_path) if f.is_relative_to(self.base_path) else f),
                    exists=True,
                    is_file=f.is_file(),
                    is_dir=f.is_dir(),
                    size=f.stat().st_size if f.is_file() else 0,
                    extension=f.suffix
                )
                results.append(info)
        except Exception:
            pass

        return results

    def file_info(self, path: str) -> FileInfo:
        """Get information about a file or directory."""
        file_path = self._resolve_path(path)

        return FileInfo(
            path=path,
            exists=file_path.exists(),
            is_file=file_path.is_file() if file_path.exists() else False,
            is_dir=file_path.is_dir() if file_path.exists() else False,
            size=file_path.stat().st_size if file_path.is_file() else 0,
            extension=file_path.suffix
        )

    def search_files(self, query: str, path: str = ".", extensions: list[str] = None) -> list[tuple[str, int, str]]:
        """
        Search for text in files.

        Args:
            query: Text to search for
            path: Directory to search in
            extensions: File extensions to search (e.g., ['.php', '.blade.php'])

        Returns:
            List of (file_path, line_number, line_content) tuples
        """
        dir_path = self._resolve_path(path)
        results = []

        if not dir_path.exists() or not dir_path.is_dir():
            return results

        extensions = extensions or ['.php', '.blade.php', '.js', '.vue', '.ts', '.json', '.env']

        for file_path in dir_path.rglob("*"):
            if not file_path.is_file():
                continue

            if extensions and file_path.suffix not in extensions:
                continue

            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    for line_num, line in enumerate(f, 1):
                        if query.lower() in line.lower():
                            rel_path = str(file_path.relative_to(self.base_path) if file_path.is_relative_to(self.base_path) else file_path)
                            results.append((rel_path, line_num, line.strip()))

                            if len(results) >= 50:  # Limit results
                                return results
            except Exception:
                continue

        return results

    def run(self, action: str, **kwargs) -> str:
        """
        Run a file operation.

        Args:
            action: One of 'read', 'write', 'list', 'info', 'search'
            **kwargs: Arguments for the action

        Returns:
            Formatted result string
        """
        if action == "read":
            return self.read_file(kwargs.get('path', ''), kwargs.get('max_lines'))

        elif action == "write":
            return self.write_file(
                kwargs.get('path', ''),
                kwargs.get('content', ''),
                kwargs.get('create_dirs', True)
            )

        elif action == "list":
            files = self.list_directory(
                kwargs.get('path', '.'),
                kwargs.get('pattern', '*'),
                kwargs.get('recursive', False)
            )
            if not files:
                return "No files found."

            output = []
            for f in files[:100]:  # Limit output
                icon = "📁" if f.is_dir else "📄"
                size = f"{f.size:,} bytes" if f.is_file else ""
                output.append(f"{icon} {f.path} {size}")
            return "\n".join(output)

        elif action == "info":
            info = self.file_info(kwargs.get('path', ''))
            return f"Path: {info.path}\nExists: {info.exists}\nType: {'directory' if info.is_dir else 'file' if info.is_file else 'unknown'}\nSize: {info.size:,} bytes\nExtension: {info.extension}"

        elif action == "search":
            results = self.search_files(
                kwargs.get('query', ''),
                kwargs.get('path', '.'),
                kwargs.get('extensions')
            )
            if not results:
                return "No matches found."

            output = []
            for path, line_num, content in results:
                output.append(f"{path}:{line_num}: {content[:100]}")
            return "\n".join(output)

        else:
            return f"Unknown action: {action}. Use 'read', 'write', 'list', 'info', or 'search'."

    def to_schema(self) -> dict:
        """Return tool schema for LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "write", "list", "info", "search"],
                        "description": "The operation to perform"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory path"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write (for 'write' action)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (for 'search' action)"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (for 'list' action)",
                        "default": "*"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Search recursively (for 'list' action)",
                        "default": False
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Max lines to read (for 'read' action)"
                    }
                },
                "required": ["action"]
            }
        }


if __name__ == "__main__":
    # Test the tool
    tool = FileOpsTool()
    print(tool.run("list", path=".", pattern="*.py"))
