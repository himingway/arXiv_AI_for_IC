"""
PDF Processing module for full-text extraction.
Downloads PDFs and extracts key sections for deep analysis.
"""

import os
import re
import requests
import tempfile
import shutil
from typing import Optional, List, Dict
import fitz  # PyMuPDF

from .database import Paper


class PDFProcessor:
    """Processes ArXiv PDFs: downloads and extracts key sections."""

    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir
        self.download_timeout = float(os.getenv('TIMEOUT_DOWNLOAD', '120'))
        os.makedirs(pdf_dir, exist_ok=True)

    def get_pdf_path(self, paper_id: str) -> str:
        """Get the local path for a paper's PDF."""
        # Sanitize paper_id to prevent path traversal
        safe_id = re.sub(r'[^a-zA-Z0-9._-]', '_', paper_id)
        return os.path.join(self.pdf_dir, f"{safe_id}.pdf")

    def is_downloaded(self, paper_id: str) -> bool:
        """Check if PDF is already downloaded."""
        return os.path.exists(self.get_pdf_path(paper_id))

    def download_pdf(self, paper: Paper) -> bool:
        """Download PDF from ArXiv."""
        pdf_path = self.get_pdf_path(paper.id)

        if self.is_downloaded(paper.id):
            return True

        try:
            response = requests.get(paper.pdf_url, timeout=self.download_timeout)
            response.raise_for_status()

            # Write atomically: temp file → rename to avoid partial downloads
            tmp_fd, tmp_path = tempfile.mkstemp(dir=self.pdf_dir, suffix='.pdf.tmp')
            try:
                with os.fdopen(tmp_fd, 'wb') as f:
                    f.write(response.content)
                shutil.move(tmp_path, pdf_path)
            except Exception:
                os.unlink(tmp_path)
                raise

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
                blocks = page.get_text("blocks")
                for b in blocks:
                    # b[6] is block_type: 0 means text block
                    if b[6] == 0:
                        text = b[4].strip()
                        if not text:
                            continue
                        
                        # Filter out the vertical arXiv watermark on the left margin
                        if text.startswith("arXiv:") and len(text) < 100:
                            continue
                        
                        # Filter out standalone page numbers
                        if text.isdigit() and len(text) <= 3:
                            continue
                            
                        full_text += text + "\n\n"
                        
            doc.close()
            return full_text

        except Exception as e:
            print(f"Error extracting text from {paper_id}: {e}")
            return None

    def get_or_download(self, paper: Paper) -> Optional[str]:
        """Download if not exists, then extract the full text."""
        if not self.download_pdf(paper):
            return None
        return self.extract_text(paper.id)

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
