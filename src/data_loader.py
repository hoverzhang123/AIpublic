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
        "# " (Level 1 heading). Each section contains the heading and its associated
        body content. This enables fine-grained retrieval where each heading+content
        block can be indexed and retrieved independently.

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
        if "# " not in text:
            return [text]

        sections: List[str] = []
        parts = text.split("# ")

        for index, part in enumerate(parts):
            if index == 0:
                if part.strip():
                    sections.append(part.strip())
                continue

            lines = part.splitlines()
            heading = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            section_text = heading if not body else f"{heading}\n{body}"
            sections.append(section_text.strip())

        return sections
