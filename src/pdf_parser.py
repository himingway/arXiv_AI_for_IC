"""
PDF Processing module for full-text extraction.
Downloads PDFs and extracts key sections for deep analysis.
"""

import json
import os
import re
import requests
import tempfile
import shutil
from typing import Optional, List, Dict, Any
import fitz  # PyMuPDF

from .database import Paper


class PDFProcessor:
    """Processes ArXiv PDFs: downloads and extracts key sections."""

    FIGURE_CAPTION_RE = re.compile(r'^(figure|fig\.?)(?:\s+|\.)*([0-9ivxlcdm]+[a-z]?)\s*:', re.IGNORECASE)

    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir
        self.figure_dir = os.path.join(pdf_dir, '_figures')
        self.download_timeout = float(os.getenv('TIMEOUT_DOWNLOAD', '120'))
        self.max_figures_per_paper = int(os.getenv('MAX_FIGURES_PER_PAPER', '4'))
        self.max_figure_scan_pages = int(os.getenv('MAX_FIGURE_SCAN_PAGES', '12'))
        os.makedirs(pdf_dir, exist_ok=True)
        os.makedirs(self.figure_dir, exist_ok=True)

    def get_pdf_path(self, paper_id: str) -> str:
        """Get the local path for a paper's PDF."""
        # Sanitize paper_id to prevent path traversal
        safe_id = re.sub(r'[^a-zA-Z0-9._-]', '_', paper_id)
        return os.path.join(self.pdf_dir, f"{safe_id}.pdf")

    def get_figure_base_dir(self, paper_id: str) -> str:
        """Get the local directory for extracted figures."""
        safe_id = re.sub(r'[^a-zA-Z0-9._-]', '_', paper_id)
        return os.path.join(self.figure_dir, safe_id)

    def get_figure_manifest_path(self, paper_id: str) -> str:
        """Get the manifest path for extracted figure metadata."""
        return os.path.join(self.get_figure_base_dir(paper_id), 'manifest.json')

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

    def _normalize_caption_text(self, text: str) -> str:
        """Normalize caption text for prompt/display use."""
        normalized = ' '.join(part.strip() for part in text.splitlines() if part.strip())
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized[:280]

    def _load_cached_figures(self, paper_id: str) -> List[Dict[str, Any]]:
        """Load cached figures from manifest if they still exist on disk."""
        manifest_path = self.get_figure_manifest_path(paper_id)
        if not os.path.exists(manifest_path):
            return []

        try:
            with open(manifest_path, 'r', encoding='utf-8') as manifest_file:
                cached_figures = json.load(manifest_file)
        except Exception:
            return []

        if not isinstance(cached_figures, list):
            return []

        valid_figures = []
        for figure in cached_figures:
            if not isinstance(figure, dict):
                continue
            image_path = figure.get('image_path')
            if image_path and os.path.exists(image_path):
                valid_figures.append(figure)
        return valid_figures

    def _save_cached_figures(self, paper_id: str, figures: List[Dict[str, Any]]) -> None:
        """Persist extracted figure metadata for reuse across reruns."""
        figure_dir = self.get_figure_base_dir(paper_id)
        os.makedirs(figure_dir, exist_ok=True)
        manifest_path = self.get_figure_manifest_path(paper_id)
        with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
            json.dump(figures, manifest_file, ensure_ascii=False, indent=2)

    def _extract_caption_blocks(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Find text blocks that look like figure captions."""
        caption_blocks = []
        for block in page.get_text('blocks'):
            if block[6] != 0:
                continue

            text = self._normalize_caption_text(block[4])
            if not text or not self.FIGURE_CAPTION_RE.match(text):
                continue

            caption_blocks.append({
                'text': text,
                'rect': fitz.Rect(block[:4]),
            })

        return caption_blocks

    def _extract_page_images(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Collect sizeable image rectangles from a page."""
        images = []
        for image in page.get_images(full=True):
            xref = image[0]
            rects = page.get_image_rects(xref)
            for rect_index, rect in enumerate(rects):
                if rect.width < 80 or rect.height < 80:
                    continue

                area = rect.width * rect.height
                if area < 20000:
                    continue

                images.append({
                    'xref': xref,
                    'rect_index': rect_index,
                    'rect': rect,
                    'area': area,
                })

        return sorted(images, key=lambda item: item['area'], reverse=True)

    def _match_caption_to_image(self, caption_block: Dict[str, Any], images: List[Dict[str, Any]], used_keys: set) -> Optional[Dict[str, Any]]:
        """Match a caption block to the nearest plausible image on the same page."""
        caption_rect = caption_block['rect']
        best_match = None
        best_score = None

        for image in images:
            image_key = (image['xref'], image['rect_index'])
            if image_key in used_keys:
                continue

            image_rect = image['rect']
            horizontal_overlap = min(caption_rect.x1, image_rect.x1) - max(caption_rect.x0, image_rect.x0)
            horizontal_gap = 0 if horizontal_overlap >= 0 else abs(horizontal_overlap)

            if caption_rect.y0 >= image_rect.y1:
                vertical_gap = caption_rect.y0 - image_rect.y1
            elif image_rect.y0 >= caption_rect.y1:
                vertical_gap = (image_rect.y0 - caption_rect.y1) + 80
            else:
                vertical_gap = 9999

            if vertical_gap > 220 or horizontal_gap > 140:
                continue

            score = vertical_gap + (horizontal_gap * 2)
            if best_score is None or score < best_score:
                best_match = image
                best_score = score

        return best_match

    def _save_figure_crop(self, page: fitz.Page, rect: fitz.Rect, paper_id: str, figure_index: int, page_number: int) -> str:
        """Crop a figure region from the PDF page and save it as PNG."""
        figure_dir = self.get_figure_base_dir(paper_id)
        os.makedirs(figure_dir, exist_ok=True)

        margin = 12
        page_rect = page.rect
        clip = fitz.Rect(
            max(page_rect.x0, rect.x0 - margin),
            max(page_rect.y0, rect.y0 - margin),
            min(page_rect.x1, rect.x1 + margin),
            min(page_rect.y1, rect.y1 + margin),
        )

        output_path = os.path.join(figure_dir, f'figure_{figure_index}_page_{page_number}.png')
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)
        pixmap.save(output_path)
        return output_path

    def _build_caption_crop_rect(self, page: fitz.Page, caption_block: Dict[str, Any]) -> fitz.Rect:
        """Build a fallback crop above a caption for vector-drawn figures."""
        caption_rect = caption_block['rect']
        page_rect = page.rect
        content_margin = 24
        vertical_padding = 8
        fallback_height = min(320, page_rect.height * 0.38)

        if caption_rect.width >= page_rect.width * 0.55:
            x0 = page_rect.x0 + content_margin
            x1 = page_rect.x1 - content_margin
        else:
            center_x = (caption_rect.x0 + caption_rect.x1) / 2
            half_width = max(page_rect.width * 0.22, caption_rect.width * 0.75)
            x0 = max(page_rect.x0 + content_margin, center_x - half_width)
            x1 = min(page_rect.x1 - content_margin, center_x + half_width)

        y1 = max(page_rect.y0 + content_margin + 40, caption_rect.y0 - vertical_padding)
        y0 = max(page_rect.y0 + content_margin, y1 - fallback_height)
        return fitz.Rect(x0, y0, x1, y1)

    def extract_figures(self, paper: Paper, max_figures: Optional[int] = None) -> List[Dict[str, Any]]:
        """Extract candidate paper figures and their captions for tweet illustration."""
        figure_limit = max_figures or self.max_figures_per_paper
        cached_figures = self._load_cached_figures(paper.id)
        if cached_figures:
            return cached_figures[:figure_limit]

        if not self.download_pdf(paper):
            return []

        pdf_path = self.get_pdf_path(paper.id)
        try:
            document = fitz.open(pdf_path)
        except Exception as exc:
            print(f"Error opening PDF for figure extraction {paper.id}: {exc}")
            return []

        figures = []
        used_image_keys = set()

        try:
            for page_number, page in enumerate(document, 1):
                if page_number > self.max_figure_scan_pages or len(figures) >= figure_limit:
                    break

                caption_blocks = self._extract_caption_blocks(page)
                if not caption_blocks:
                    continue

                images = self._extract_page_images(page)

                for caption_block in caption_blocks:
                    if len(figures) >= figure_limit:
                        break

                    matched_image = self._match_caption_to_image(caption_block, images, used_image_keys)
                    if matched_image is not None:
                        crop_rect = matched_image['rect']
                        used_image_keys.add((matched_image['xref'], matched_image['rect_index']))
                    else:
                        crop_rect = self._build_caption_crop_rect(page, caption_block)

                    figure_index = len(figures) + 1
                    image_path = self._save_figure_crop(
                        page,
                        crop_rect,
                        paper.id,
                        figure_index,
                        page_number,
                    )

                    figures.append({
                        'figure_key': f'图{figure_index}',
                        'caption': caption_block['text'],
                        'image_path': image_path,
                        'page': page_number,
                    })
        finally:
            document.close()

        if figures:
            self._save_cached_figures(paper.id, figures)

        return figures

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
