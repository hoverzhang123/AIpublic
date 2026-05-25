from pathlib import Path
from typing import List


class DataLoader:
    def __init__(self, source_directory: Path):
        self.source_directory = source_directory

    def load_documents(self) -> List[str]:
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
