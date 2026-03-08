"""
ZIMA.md Instruction System

Inspired by OpenCode's CLAUDE.md system.
Allows users to customize agent behavior per-project or globally.

Search order (later files override earlier):
1. ~/.config/zima/ZIMA.md          (global user preferences)
2. {project}/.zima/ZIMA.md         (project workspace)
3. {project}/ZIMA.md               (project root)

Instructions are injected into the system prompt before each request.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class InstructionSource:
    """Represents a loaded instruction file."""
    path: Path
    content: str
    level: str  # 'global', 'workspace', 'project'


class InstructionLoader:
    """
    Loads ZIMA.md instruction files from multiple locations.

    Usage:
        loader = InstructionLoader(project_dir="/path/to/project")
        instructions = loader.load_all()
        # Returns merged instructions string
    """

    FILENAME = "ZIMA.md"
    ALT_FILENAMES = ["zima.md", "ZIMA.MD"]  # Case variations

    def __init__(self, project_dir: Optional[str] = None):
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.sources: list[InstructionSource] = []

    def _find_file(self, directory: Path) -> Optional[Path]:
        """Find instruction file in directory (case-insensitive)."""
        for filename in [self.FILENAME] + self.ALT_FILENAMES:
            filepath = directory / filename
            if filepath.exists() and filepath.is_file():
                return filepath
        return None

    def _load_file(self, filepath: Path, level: str) -> Optional[InstructionSource]:
        """Load a single instruction file."""
        try:
            content = filepath.read_text(encoding='utf-8').strip()
            if content:
                return InstructionSource(
                    path=filepath,
                    content=content,
                    level=level
                )
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
        return None

    def _get_global_dir(self) -> Path:
        """Get global config directory."""
        # Check XDG_CONFIG_HOME first, then fallback to ~/.config
        xdg_config = os.environ.get('XDG_CONFIG_HOME')
        if xdg_config:
            return Path(xdg_config) / 'zima'
        return Path.home() / '.config' / 'zima'

    def load_all(self) -> str:
        """
        Load all instruction files and merge them.

        Returns:
            Merged instructions string with source comments.
        """
        self.sources = []

        # 1. Global: ~/.config/zima/ZIMA.md
        global_dir = self._get_global_dir()
        global_file = self._find_file(global_dir)
        if global_file:
            source = self._load_file(global_file, 'global')
            if source:
                self.sources.append(source)

        # 2. Workspace: {project}/.zima/ZIMA.md
        workspace_dir = self.project_dir / '.zima'
        workspace_file = self._find_file(workspace_dir)
        if workspace_file:
            source = self._load_file(workspace_file, 'workspace')
            if source:
                self.sources.append(source)

        # 3. Project root: {project}/ZIMA.md
        project_file = self._find_file(self.project_dir)
        if project_file:
            source = self._load_file(project_file, 'project')
            if source:
                self.sources.append(source)

        return self._merge_instructions()

    def _merge_instructions(self) -> str:
        """Merge all loaded instructions with source headers."""
        if not self.sources:
            return ""

        parts = ["USER INSTRUCTIONS:"]

        for source in self.sources:
            # Add source header for debugging
            relative_path = source.path
            try:
                relative_path = source.path.relative_to(Path.home())
                relative_path = f"~/{relative_path}"
            except ValueError:
                pass

            parts.append(f"\n# From {relative_path} ({source.level}):")
            parts.append(source.content)

        return "\n".join(parts)

    def get_sources(self) -> list[InstructionSource]:
        """Get list of loaded instruction sources."""
        return self.sources

    def has_instructions(self) -> bool:
        """Check if any instructions were loaded."""
        return len(self.sources) > 0


def load_instructions(project_dir: Optional[str] = None) -> str:
    """
    Convenience function to load all instructions.

    Args:
        project_dir: Project directory path (defaults to cwd)

    Returns:
        Merged instructions string, or empty string if none found.
    """
    loader = InstructionLoader(project_dir)
    return loader.load_all()


# Template for new ZIMA.md files
ZIMA_TEMPLATE = """# ZIMA.md - Project Instructions

This file provides instructions to Zima (the AI coding assistant).
Add project-specific rules, preferences, and context here.

## Project Overview
<!-- Describe what this project is -->


## Code Style
<!-- Add coding conventions -->
-

## Commands
<!-- Common commands to run -->
- `npm test` - Run tests
- `npm run build` - Build project

## Important Notes
<!-- Gotchas, warnings, special considerations -->
-

## File Structure
<!-- Key directories and their purposes -->
- `src/` - Source code
- `tests/` - Test files
"""


def create_template(directory: Optional[str] = None, force: bool = False) -> Path:
    """
    Create a ZIMA.md template file.

    Args:
        directory: Where to create the file (defaults to cwd)
        force: Overwrite existing file

    Returns:
        Path to created file

    Raises:
        FileExistsError: If file exists and force=False
    """
    target_dir = Path(directory) if directory else Path.cwd()
    filepath = target_dir / "ZIMA.md"

    if filepath.exists() and not force:
        raise FileExistsError(f"ZIMA.md already exists at {filepath}")

    filepath.write_text(ZIMA_TEMPLATE, encoding='utf-8')
    return filepath


if __name__ == "__main__":
    # Test the instruction loader
    print("Testing InstructionLoader...")
    print(f"Project dir: {Path.cwd()}")

    loader = InstructionLoader()
    instructions = loader.load_all()

    if loader.has_instructions():
        print(f"\nFound {len(loader.sources)} instruction file(s):")
        for source in loader.sources:
            print(f"  - {source.path} ({source.level})")
        print(f"\nMerged instructions ({len(instructions)} chars):")
        print(instructions[:500] + "..." if len(instructions) > 500 else instructions)
    else:
        print("\nNo instruction files found.")
        print("Create one with: python instructions.py --create")

    # Handle --create flag
    import sys
    if "--create" in sys.argv:
        try:
            path = create_template()
            print(f"\nCreated template at: {path}")
        except FileExistsError as e:
            print(f"\n{e}")
