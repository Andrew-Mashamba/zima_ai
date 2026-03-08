"""
Repository Map Tool - Inspired by Aider's approach.

Creates a concise summary of the codebase structure using AST parsing.
Gives the LLM awareness of what actually exists without reading every file.
"""

import ast
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CodeSymbol:
    """Represents a code symbol (function, class, method)."""
    name: str
    kind: str  # 'function', 'class', 'method'
    signature: str
    file_path: str
    line_number: int


@dataclass
class FileMap:
    """Map of a single file's structure."""
    path: str
    language: str
    symbols: list[CodeSymbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)


class RepoMapTool:
    """
    Creates a concise map of repository structure.

    Unlike just listing files, this extracts:
    - Function signatures
    - Class definitions
    - Method signatures
    - Import statements

    This gives the LLM real awareness of what exists in the codebase.
    """

    name = "repo_map"
    description = "Get a structural map of the codebase showing functions, classes, and their signatures"

    # File extensions to parse
    PARSEABLE = {
        '.py': 'python',
        '.php': 'php',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
    }

    # Directories to skip
    SKIP_DIRS = {
        '__pycache__', 'node_modules', '.git', '.venv', 'venv',
        'vendor', 'dist', 'build', '.next', 'storage', 'bootstrap/cache'
    }

    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()

    def _parse_python_file(self, file_path: Path) -> FileMap:
        """Parse a Python file and extract symbols."""
        file_map = FileMap(
            path=str(file_path.relative_to(self.base_path)),
            language='python'
        )

        try:
            content = file_path.read_text(encoding='utf-8', errors='replace')
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError):
            return file_map

        for node in ast.walk(tree):
            # Extract imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    file_map.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    file_map.imports.append(f"{module}.{alias.name}")

            # Extract function definitions
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Skip private/dunder methods for brevity
                if node.name.startswith('_') and not node.name.startswith('__init__'):
                    continue

                # Build signature
                args = []
                for arg in node.args.args:
                    arg_str = arg.arg
                    if arg.annotation:
                        try:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        except:
                            pass
                    args.append(arg_str)

                # Return type
                return_type = ""
                if node.returns:
                    try:
                        return_type = f" -> {ast.unparse(node.returns)}"
                    except:
                        pass

                sig = f"def {node.name}({', '.join(args)}){return_type}"

                file_map.symbols.append(CodeSymbol(
                    name=node.name,
                    kind='function',
                    signature=sig,
                    file_path=str(file_path.relative_to(self.base_path)),
                    line_number=node.lineno
                ))

            # Extract class definitions
            elif isinstance(node, ast.ClassDef):
                # Get base classes
                bases = []
                for base in node.bases:
                    try:
                        bases.append(ast.unparse(base))
                    except:
                        pass

                base_str = f"({', '.join(bases)})" if bases else ""
                sig = f"class {node.name}{base_str}"

                file_map.symbols.append(CodeSymbol(
                    name=node.name,
                    kind='class',
                    signature=sig,
                    file_path=str(file_path.relative_to(self.base_path)),
                    line_number=node.lineno
                ))

                # Extract methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name.startswith('_') and item.name != '__init__':
                            continue

                        args = []
                        for arg in item.args.args:
                            if arg.arg == 'self':
                                continue
                            arg_str = arg.arg
                            if arg.annotation:
                                try:
                                    arg_str += f": {ast.unparse(arg.annotation)}"
                                except:
                                    pass
                            args.append(arg_str)

                        sig = f"  def {item.name}({', '.join(args)})"

                        file_map.symbols.append(CodeSymbol(
                            name=f"{node.name}.{item.name}",
                            kind='method',
                            signature=sig,
                            file_path=str(file_path.relative_to(self.base_path)),
                            line_number=item.lineno
                        ))

        return file_map

    def _should_skip(self, path: Path) -> bool:
        """Check if path should be skipped."""
        for part in path.parts:
            if part in self.SKIP_DIRS or part.startswith('.'):
                return True
        return False

    def build_map(self, max_tokens: int = 2000) -> str:
        """
        Build a repository map within token budget.

        Args:
            max_tokens: Approximate token limit (chars / 4)

        Returns:
            Formatted string map of the repository
        """
        file_maps: list[FileMap] = []

        # Find all parseable files
        for ext, lang in self.PARSEABLE.items():
            for file_path in self.base_path.rglob(f"*{ext}"):
                if self._should_skip(file_path):
                    continue

                if lang == 'python':
                    file_map = self._parse_python_file(file_path)
                    if file_map.symbols:  # Only include files with symbols
                        file_maps.append(file_map)

        # Sort by number of symbols (most important first)
        file_maps.sort(key=lambda x: len(x.symbols), reverse=True)

        # Build output within token budget
        output_lines = ["# Repository Structure\n"]
        char_budget = max_tokens * 4  # Rough chars-to-tokens ratio
        current_chars = len(output_lines[0])

        for file_map in file_maps:
            file_section = [f"\n## {file_map.path}\n"]

            for symbol in file_map.symbols:
                file_section.append(f"{symbol.signature}\n")

            section_text = ''.join(file_section)
            if current_chars + len(section_text) > char_budget:
                break

            output_lines.extend(file_section)
            current_chars += len(section_text)

        return ''.join(output_lines)

    def run(self, max_tokens: int = 2000) -> str:
        """Run the repo map tool."""
        try:
            return self.build_map(max_tokens)
        except Exception as e:
            return f"Error building repo map: {e}"

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens for the map (default 2000)"
                    }
                }
            }
        }


if __name__ == "__main__":
    # Test the repo map
    tool = RepoMapTool()
    print(tool.run())
