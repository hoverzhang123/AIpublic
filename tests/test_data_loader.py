"""
Unit tests for src/data_loader.py - Document loading and parsing.

Tests the DataLoader class for loading markdown files, splitting sections,
filtering empty documents, and handling edge cases.
"""

from pathlib import Path

import pytest

from src.data_loader import DataLoader


class TestDataLoaderInitialization:
    """Tests for DataLoader initialization."""

    def test_dataloader_init_stores_directory(self, tmp_path):
        """DataLoader stores the source directory."""
        loader = DataLoader(tmp_path)
        assert loader.source_directory == tmp_path


class TestLoadDocuments:
    """Tests for DataLoader.load_documents() method."""

    def test_load_documents_with_single_file(self, tmp_path):
        """load_documents() loads and parses a single markdown file."""
        # Setup: create markdown file
        md_file = tmp_path / "test.md"
        md_file.write_text("# Heading 1\nContent 1\n# Heading 2\nContent 2")

        # Load
        loader = DataLoader(tmp_path)
        docs = loader.load_documents()

        # Verify
        assert len(docs) == 2
        assert "Heading 1" in docs[0]
        assert "Content 1" in docs[0]
        assert "Heading 2" in docs[1]
        assert "Content 2" in docs[1]

    def test_load_documents_with_multiple_files(self, tmp_path):
        """load_documents() loads and parses multiple markdown files."""
        # Setup: create multiple files
        (tmp_path / "file1.md").write_text("# Doc 1\nContent 1")
        (tmp_path / "file2.md").write_text("# Doc 2\nContent 2")

        # Load
        loader = DataLoader(tmp_path)
        docs = loader.load_documents()

        # Verify: should have at least 2 sections
        assert len(docs) >= 2
        texts = " ".join(docs)
        assert "Doc 1" in texts
        assert "Doc 2" in texts
        assert "Content 1" in texts
        assert "Content 2" in texts

    def test_load_documents_from_subdirectories(self, tmp_path):
        """load_documents() recursively discovers files in subdirectories."""
        # Setup: create nested structure
        sub_dir = tmp_path / "subfolder"
        sub_dir.mkdir()
        (tmp_path / "root.md").write_text("# Root\nRoot content")
        (sub_dir / "nested.md").write_text("# Nested\nNested content")

        # Load
        loader = DataLoader(tmp_path)
        docs = loader.load_documents()

        # Verify
        texts = " ".join(docs)
        assert "Root" in texts
        assert "Nested" in texts

    def test_load_documents_filters_empty_documents(self, tmp_path):
        """load_documents() filters out empty documents."""
        # Setup: file with empty content
        (tmp_path / "file1.md").write_text("# Heading 1\nContent")
        (tmp_path / "empty.md").write_text("")
        (tmp_path / "file2.md").write_text("# Heading 2\n")

        # Load
        loader = DataLoader(tmp_path)
        docs = loader.load_documents()

        # Verify: empty file should be filtered, "# Heading 2" with no body should be kept
        assert len([d for d in docs if d.strip()]) > 0
        for doc in docs:
            assert doc.strip()  # All docs should be non-empty after filtering

    def test_load_documents_raises_on_nonexistent_directory(self):
        """load_documents() raises FileNotFoundError for missing directory."""
        loader = DataLoader(Path("/nonexistent/path/to/docs"))

        with pytest.raises(FileNotFoundError, match="Source directory does not exist"):
            loader.load_documents()

    def test_load_documents_raises_on_no_markdown_files(self, tmp_path):
        """load_documents() raises FileNotFoundError when no .md files found."""
        # Setup: create directory with no markdown files
        (tmp_path / "file.txt").write_text("Not markdown")

        loader = DataLoader(tmp_path)

        with pytest.raises(FileNotFoundError, match="No markdown documents found"):
            loader.load_documents()

    def test_load_documents_returns_list_of_strings(self, tmp_path):
        """load_documents() returns a list of strings."""
        (tmp_path / "test.md").write_text("# Heading\nContent")

        loader = DataLoader(tmp_path)
        docs = loader.load_documents()

        assert isinstance(docs, list)
        assert all(isinstance(doc, str) for doc in docs)

    def test_load_documents_sorts_files_by_path(self, tmp_path):
        """load_documents() processes files in sorted order."""
        # Setup: create files with predictable order
        (tmp_path / "aaa.md").write_text("# AAA\nContent")
        (tmp_path / "zzz.md").write_text("# ZZZ\nContent")
        (tmp_path / "bbb.md").write_text("# BBB\nContent")

        loader = DataLoader(tmp_path)
        docs = loader.load_documents()

        # AAA should come before ZZZ and BBB
        combined = " ".join(docs)
        aaa_idx = combined.index("AAA")
        bbb_idx = combined.index("BBB")
        zzz_idx = combined.index("ZZZ")

        assert aaa_idx < bbb_idx < zzz_idx


class TestSplitMarkdownSections:
    """Tests for DataLoader._split_markdown_sections() static method."""

    def test_split_markdown_with_single_heading(self, sample_markdown_single_section):
        """_split_markdown_sections() returns text as-is if no headings present."""
        result = DataLoader._split_markdown_sections(sample_markdown_single_section)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == sample_markdown_single_section

    def test_split_markdown_with_multiple_headings(
        self, sample_markdown_multiple_sections
    ):
        """_split_markdown_sections() splits at level-1 headings."""
        result = DataLoader._split_markdown_sections(sample_markdown_multiple_sections)

        assert len(result) == 3
        # Each section should contain heading and content
        assert any("Section 1" in s for s in result)
        assert any("Section 2" in s for s in result)
        assert any("Section 3" in s for s in result)

    def test_split_markdown_preserves_heading_and_body(
        self, sample_markdown_multiple_sections
    ):
        """_split_markdown_sections() preserves heading + body structure."""
        result = DataLoader._split_markdown_sections(sample_markdown_multiple_sections)

        # First section should contain heading and its content
        first = result[0]
        assert "Section 1" in first
        assert "content of section 1" in first

    def test_split_markdown_with_subheadings(self, sample_markdown_mixed_levels):
        """_split_markdown_sections() only splits on level-1 headings (# )."""
        result = DataLoader._split_markdown_sections(sample_markdown_mixed_levels)

        # Should have 2 sections (only # splits, not ## )
        assert len(result) == 2
        # Subsection should be kept with parent
        assert any("Subsection" in s for s in result)

    def test_split_markdown_with_empty_sections(self):
        """_split_markdown_sections() handles heading with no body."""
        text = "# Heading 1\n# Heading 2\nContent"
        result = DataLoader._split_markdown_sections(text)

        assert len(result) == 2
        assert "Heading 1" in result[0]
        # Heading 1 has no body, just the heading
        assert result[0].strip() == "Heading 1"

    def test_split_markdown_with_multiline_body(self):
        """_split_markdown_sections() preserves multiline content."""
        text = "# Heading\nLine 1\nLine 2\nLine 3"
        result = DataLoader._split_markdown_sections(text)

        assert len(result) == 1
        assert "Heading" in result[0]
        assert "Line 1" in result[0]
        assert "Line 2" in result[0]
        assert "Line 3" in result[0]

    def test_split_markdown_with_preamble(self):
        """_split_markdown_sections() includes preamble before first heading."""
        text = "Preamble text here\n# Heading 1\nContent"
        result = DataLoader._split_markdown_sections(text)

        # Should have 2 sections: preamble + heading+content
        assert len(result) == 2
        assert "Preamble" in result[0]
        assert "Heading 1" in result[1]

    def test_split_markdown_strips_whitespace(self):
        """_split_markdown_sections() strips leading/trailing whitespace from sections."""
        text = "\n\n# Heading\n\nContent\n\n"
        result = DataLoader._split_markdown_sections(text)

        for section in result:
            assert not section.startswith("\n")
            assert not section.endswith("\n")

    def test_split_markdown_empty_input(self):
        """_split_markdown_sections() handles empty string."""
        result = DataLoader._split_markdown_sections("")

        assert result == [""]

    def test_split_markdown_only_headings(self):
        """_split_markdown_sections() handles markdown with only headings."""
        text = "# Heading 1\n# Heading 2\n# Heading 3"
        result = DataLoader._split_markdown_sections(text)

        assert len(result) == 3
        assert all("Heading" in s for s in result)


class TestDataLoaderEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_load_documents_with_unicode_content(self, tmp_path):
        """load_documents() handles Unicode content correctly."""
        # Create file with Unicode characters (explicit UTF-8 encoding for Windows compatibility)
        (tmp_path / "unicode.md").write_text(
            "# Unicode Test\nHello World 🌍\nMultilingual content", encoding="utf-8"
        )

        loader = DataLoader(tmp_path)
        docs = loader.load_documents()

        assert len(docs) > 0
        combined = " ".join(docs)
        assert "Hello" in combined
        assert "🌍" in combined
        assert "Multilingual" in combined

    def test_load_documents_with_windows_line_endings(self, tmp_path):
        """load_documents() handles Windows line endings."""
        # Use \r\n line endings
        (tmp_path / "windows.md").write_text(
            "# Heading 1\r\nContent 1\r\n# Heading 2\r\nContent 2", newline=""
        )

        loader = DataLoader(tmp_path)
        docs = loader.load_documents()

        assert len(docs) == 2

    def test_load_documents_with_large_file(self, tmp_path):
        """load_documents() can handle reasonably large files."""
        # Create a file with many sections
        content = "\n".join([f"# Section {i}\nContent {i}" for i in range(100)])
        (tmp_path / "large.md").write_text(content)

        loader = DataLoader(tmp_path)
        docs = loader.load_documents()

        assert len(docs) == 100

    def test_split_markdown_with_code_blocks(self):
        """_split_markdown_sections() preserves code blocks with # in them."""
        text = """# Introduction
Some text.
```python
# This is a comment in code
def foo():
    pass
```
More text.
# Next Section
Content."""
        result = DataLoader._split_markdown_sections(text)

        # Should only split on the # that starts a line (markdown heading)
        assert len(result) == 2
        # Code block should be preserved in first section
        assert "def foo" in result[0]

    def test_load_documents_returns_stripped_sections(self, tmp_path):
        """load_documents() returns stripped sections without extra whitespace."""
        (tmp_path / "test.md").write_text("\n\n# Heading\n\nContent\n\n")

        loader = DataLoader(tmp_path)
        docs = loader.load_documents()

        for doc in docs:
            assert doc == doc.strip()
