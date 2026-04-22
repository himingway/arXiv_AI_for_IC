"""
Deep Article Synthesis module.
Generates in-depth technical blog posts/tweets from selected papers.
Supports both OpenAI-compatible APIs and Anthropic Claude API.
"""

import os
import datetime
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import anthropic

from .database import Paper, Database
from .pdf_parser import PDFProcessor


# Load environment variables from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / '.env')


class DeepSynthesizer:
    """Generates deep technical analysis articles from selected papers.
    Supports both OpenAI-compatible APIs and Anthropic Claude API.
    """


    SYSTEM_PROMPT = """你是一位资深 SoC 与一致性互联专家。请基于提供的论文信息与全文，输出一篇面向芯片硬件设计工程师的深度技术推文。重点讲解论文内容，并点评（不要出现专家/架构师等字眼，要低调）。
"""

    USER_PROMPT_TEMPLATE = """论文信息：
标题：{title}
作者：{authors}
AI评分：{score}/10
标签：{tags}
摘要：{abstract}

论文全文文本：
---
{full_text}
---

请按“公众号/推文可读性优先”的方式组织内容，保证结构清晰但标题表达可以自然变化。不要输出无关免责声明。"""

    def __init__(self):
        self.provider = os.getenv('LLM_PROVIDER', 'openai').lower()
        self.api_key = os.getenv('API_KEY', '')
        self.model = os.getenv('LLM_MODEL', 'gpt-4o')
        self.temperature = float(os.getenv('TEMPERATURE', '0.3'))
        self.max_tokens = int(os.getenv('MAX_TOKENS_SYNTHESIS', '8192'))
        self.timeout = float(os.getenv('TIMEOUT_SYNTHESIS', '300'))  # 5 minutes default timeout

        if self.provider == 'openai':
            self.base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout
            )
        elif self.provider == 'anthropic':
            self.base_url = os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')
            self.client = anthropic.Anthropic(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}. Use 'openai' or 'anthropic'.")

    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key and self.api_key != 'your_api_key_here')

    def synthesize_paper(self, paper: Paper, full_text: str) -> Optional[str]:
        """Generate deep synthesis for a single paper."""
        if not self.is_configured():
            return None

        prompt = self.USER_PROMPT_TEMPLATE.format(
            title=paper.title,
            authors=paper.authors,
            score=paper.ai_score or 'N/A',
            tags=paper.ai_tags or 'N/A',
            abstract=paper.abstract,
            full_text=full_text
        )

        try:
            if self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                content = response.choices[0].message.content.strip()

            elif self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    system=self.SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                content = response.content[0].text.strip()

            return content

        except Exception as e:
            print(f"Error synthesizing paper {paper.id}: {e}")
            return None

    def synthesize_multiple(self, papers: List[Paper], pdf_processor: PDFProcessor) -> str:
        """Generate synthesis for multiple selected papers."""
        if not self.is_configured():
            return "错误：API 密钥未配置，请检查 .env 文件。"

        result = f"# ArXiv 芯片架构前沿精选\n\n"
        result += f"生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        result += "---\n\n"

        for i, paper in enumerate(papers, 1):
            print(f"Processing paper {i}/{len(papers)}: {paper.title[:50]}...")

            # Get full text from PDF
            full_text = pdf_processor.get_or_download(paper)
            if full_text is None:
                result += f"## {i}. {paper.title}\n\n"
                result += f"**作者**: {paper.authors}\n\n"
                result += f"评分: **{paper.ai_score}/10** 标签: {paper.ai_tags}\n\n"
                result += f"PDF 下载失败，无法生成深度分析。\n\n"
                result += f"原文链接: {paper.pdf_url}\n\n"
                result += "---\n\n"
                continue

            synthesis = self.synthesize_paper(paper, full_text)
            if synthesis is None:
                result += f"## {i}. {paper.title}\n\n"
                result += f"**作者**: {paper.authors}\n\n"
                result += f"评分: **{paper.ai_score}/10** 标签: {paper.ai_tags}\n\n"
                result += f"AI 生成失败。\n\n"
                result += f"原文链接: {paper.pdf_url}\n\n"
                result += "---\n\n"
                continue

            result += f"## {i}. {paper.title}\n\n"
            result += f"**作者**: {paper.authors}\n\n"
            result += f"评分: **{paper.ai_score}/10** 标签: {paper.ai_tags}\n\n"
            result += f"ArXiv: [{paper.id}]({paper.pdf_url})\n\n"
            result += "---\n\n"
            result += synthesis
            result += "\n\n---\n\n"

        return result
