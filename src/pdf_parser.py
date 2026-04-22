"""
PDF Processing module for full-text extraction.
Downloads PDFs and extracts key sections for deep analysis.
"""

import os
import requests
import re
from typing import Optional, List, Dict
import fitz  # PyMuPDF

from .database import Paper


class PDFProcessor:
    """Processes ArXiv PDFs: downloads and extracts key sections."""

    # Section headers that indicate key content
    TARGET_SECTIONS = [
        # Architecture sections
        r'^\s*(1\s*)?architecture\b',
        r'^\s*(1\s*)?microarchitecture\b',
        r'^\s*(1\s*)?design\b',
        r'^\s*(1\s*)?implementation\b',
        r'^\s*(1\s*)?methodology\b',
        r'^\s*(1\s*)?method\b',
        # Hardware organization
        r'^\s*(2\s*)?architecture\b',
        r'^\s*(2\s*)?microarchitecture\b',
        r'^\s*(2\s*)?design\b',
        r'^\s*(2\s*)?implementation\b',
        # Implementation details
        r'^\s*(3\s*)?implementation\b',
        r'^\s*(3\s*)?architecture\b',
        r'^\s*(3\s*)?design\b',
        # EDA specific
        r'^\s*(proposed|our)\s*method\b',
        r'^\s*(proposed|our)\s*approach\b',
        r'^\s*experiment\w*\b',
        r'^\s*evaluation\b',
        r'^\s*result\w*\b',
    ]

    END_SECTIONS = [
        r'^\s*conclusion\b',
        r'^\s*references\b',
        r'^\s*acknowledgments?\b',
        r'^\s*appendix\b',
    ]

    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir
        os.makedirs(pdf_dir, exist_ok=True)

    def get_pdf_path(self, paper_id: str) -> str:
        """Get the local path for a paper's PDF."""
        return os.path.join(self.pdf_dir, f"{paper_id}.pdf")

    def is_downloaded(self, paper_id: str) -> bool:
        """Check if PDF is already downloaded."""
        return os.path.exists(self.get_pdf_path(paper_id))

    def download_pdf(self, paper: Paper) -> bool:
        """Download PDF from ArXiv."""
        pdf_path = self.get_pdf_path(paper.id)

        if self.is_downloaded(paper.id):
            return True

        try:
            response = requests.get(paper.pdf_url, timeout=120)
            response.raise_for_status()

            with open(pdf_path, 'wb') as f:
                f.write(response.content)

            return True

        except Exception as e:
            print(f"Error downloading PDF for {paper.id}: {e}")
            return False

    def extract_text(self, paper_id: str) -> Optional[str]:
        """Extract full text from downloaded PDF."""
        pdf_path = self.get_pdf_path(paper_id)

        if not os.path.exists(pdf_path):
            return None

        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n\n"
            doc.close()
            return full_text

        except Exception as e:
            print(f"Error extracting text from {paper_id}: {e}")
            return None

    def extract_key_sections(self, paper_id: str) -> Optional[str]:
        """Extract key sections (architecture, implementation) from PDF."""
        full_text = self.extract_text(paper_id)
        if full_text is None:
            return None

        return self._extract_sections_from_text(full_text)

    def _extract_sections_from_text(self, full_text: str) -> str:
        """
        Extract relevant sections from full text.
        Uses regex matching to find architecture/implementation sections.
        """
        lines = full_text.split('\n')
        capturing = False
        captured_lines: List[str] = []
        captured_chars = 0
        max_chars = 15000  # Limit to avoid exceeding context window

        for line in lines:
            line_lower = line.lower()

            # Check if we should start capturing this section
            if not capturing:
                for pattern in self.TARGET_SECTIONS:
                    if re.search(pattern, line_lower):
                        capturing = True
                        captured_lines.append(line)
                        captured_chars += len(line)
                        break
            else:
                # Check if we've hit an ending section
                end_now = False
                for pattern in self.END_SECTIONS:
                    if re.search(pattern, line_lower):
                        end_now = True
                        break
                if end_now:
                    break

                captured_lines.append(line)
                captured_chars += len(line)

                # Stop if we've captured enough
                if captured_chars >= max_chars:
                    break

        # If we didn't capture anything (unusual section numbering), fall back to middle 50%
        if captured_chars < 500:
            # Get the text from abstract to before conclusion/references
            total_lines = len(lines)
            start = int(total_lines * 0.15)  # Skip title/abstract/intro
            end = int(total_lines * 0.75)  # Stop before references/conclusion
            captured_lines = lines[start:end]
            captured_text = '\n'.join(captured_lines)
            # Still limit length
            if len(captured_text) > max_chars:
                captured_text = captured_text[:max_chars]
            return captured_text

        captured_text = '\n'.join(captured_lines)
        if len(captured_text) > max_chars:
            captured_text = captured_text[:max_chars]

        return captured_text

    def get_or_download(self, paper: Paper) -> Optional[str]:
        """Download if not exists, then extract key sections."""
        if not self.download_pdf(paper):
            return None
        return self.extract_key_sections(paper.id)

    def delete_pdf(self, paper_id: str) -> bool:
        """Delete downloaded PDF to save space."""
        pdf_path = self.get_pdf_path(paper_id)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            return True
        return False

    def get_storage_stats(self) -> Dict[str, any]:
        """Get storage statistics."""
        if not os.path.exists(self.pdf_dir):
            return {'file_count': 0, 'total_size_bytes': 0}

        total_size = 0
        file_count = 0
        for entry in os.scandir(self.pdf_dir):
            if entry.is_file() and entry.name.endswith('.pdf'):
                total_size += entry.stat().st_size
                file_count += 1

        return {
            'file_count': file_count,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }
