from pathlib import Path
from typing import List


class DataLoader:
    """
    Loads and parses Markdown documents from a directory structure.

    Recursively discovers all Markdown files in a given directory and splits them
    into semantically meaningful sections based on Markdown heading boundaries
    (# Level 1 headings). This modular approach enables fine-grained document
    retrieval and prevents overly long documents from being stored as single entities.

    Used by the RAG pipeline to load the FAQ knowledge base before generating embeddings.
    """

    def __init__(self, source_directory: Path):
        """
        Initialize the DataLoader with a source directory path.

        Args:
            source_directory (Path): Path object pointing to the directory containing
                                    Markdown documents (.md files). This directory is
                                    scanned recursively for all Markdown files.
        """
        self.source_directory = source_directory

    def load_documents(self) -> List[str]:
        """
        Load and parse all Markdown documents from the source directory.

        Recursively discovers all .md files in the source directory (sorted by path),
        loads each file as UTF-8 text, and splits it into semantically meaningful
        sections at Markdown heading boundaries. Empty sections are filtered out.

        This method is called during RAG pipeline initialization to populate the vector
        database with indexed knowledge.

        Returns:
            List[str]: List of document sections (non-empty strings). Each section is
                      either a heading without body, or "heading\\nbody" format where
                      the section is extracted by splitting on "# " (Level 1 heading).

        Raises:
            FileNotFoundError: If the source directory does not exist or contains
                             no Markdown files (.md).

        Example:
            >>> loader = DataLoader(Path("src/milvus_docs/en/faq"))
            >>> docs = loader.load_documents()
            >>> len(docs)  # Number of sections across all markdown files
            42
        """
        if not self.source_directory.exists():
            raise FileNotFoundError(
                f"Source directory does not exist: {self.source_directory}"
            )

        markdown_files = sorted(self.source_directory.rglob("*.md"))
        if not markdown_files:
            raise FileNotFoundError(
                f"No markdown documents found in: {self.source_directory}"
            )

        documents: List[str] = []
        for markdown_path in markdown_files:
            file_text = markdown_path.read_text(encoding="utf-8")
            documents.extend(self._split_markdown_sections(file_text))

        return [document.strip() for document in documents if document.strip()]

    @staticmethod
    def _split_markdown_sections(text: str) -> List[str]:
        """
        Split Markdown text into sections at Level 1 heading boundaries.

        Parses a single Markdown document and splits it into sections at each
        "# " (Level 1 heading, not "## ", "### ", etc.). Each section contains the heading
        and its associated body content. This enables fine-grained retrieval where each
        heading+content block can be indexed and retrieved independently.

        Respects code block boundaries (lines between triple backticks) and does not split
        on "# " that appears inside code blocks or as part of higher-level headings (##, ###, etc.).

        Args:
            text (str): Raw Markdown text content from a single file.

        Returns:
            List[str]: List of document sections. If the text has no "# " headings,
                      returns [text] as a single section. Otherwise, returns one entry
                      per heading with format "heading_title" or "heading_title\\nbody".

        Example:
            >>> text = "# Section A\\nContent A\\n# Section B\\nContent B"
            >>> sections = DataLoader._split_markdown_sections(text)
            >>> len(sections)  # 2
            >>> sections[0]  # "Section A\\nContent A"
        """
        lines = text.split("\n")
        sections: List[str] = []
        current_section: List[str] = []
        in_code_block = False

        for line in lines:
            # Toggle code block state on triple backticks
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                current_section.append(line)
                continue

            # Only split on "# " at line start if not in code block and it's a level-1 heading
            if (
                not in_code_block
                and line.startswith("# ")
                and not line.startswith("## ")
            ):
                # Save current section if it has content
                if current_section:
                    section_text = "\n".join(current_section).strip()
                    if section_text:
                        sections.append(section_text)
                    current_section = []

                # Start new section with this heading (strip "# " prefix)
                heading = line[2:].strip()  # Remove "# " prefix
                current_section.append(heading)
            else:
                current_section.append(line)

        # Add final section
        if current_section:
            section_text = "\n".join(current_section).strip()
            if section_text:
                sections.append(section_text)

        return sections if sections else [text]
