"""
Deep Article Synthesis module.
Generates in-depth technical blog posts/tweets from selected papers.
Supports both OpenAI-compatible APIs and Anthropic Claude API.
"""

import os
import datetime
from typing import List, Optional, Dict, Any
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


    SYSTEM_PROMPT = """你是一位资深 SoC 与一致性互联专家。请基于提供的论文信息与全文，输出一篇面向芯片硬件设计工程师的深度技术推文。
注意：你的输出必须足够长、足够深入。绝不能仅作表面总结。你必须从硬件和架构设计的角度详细剖析论文的核心技术原理、系统架构、具体机制或算法流程，把“为什么这么做”和“具体是怎么做到的”原原本本地解释清楚。重点讲解论文内容并给出专业点评（不要出现专家/架构师等字眼，要低调）。"""

    USER_PROMPT_TEMPLATE = """论文信息：
标题：{title}
作者：{authors}
AI评分：{score}/10
标签：{tags}
摘要：{abstract}

论文候选配图：
{figures_summary}

论文全文文本：
---
{full_text}
---

请按“公众号/推文可读性优先”的方式组织内容，保证结构清晰但标题表达可以自然变化。
必须包含以下深度分析模块，且内容不仅要详实，更要把技术原理真正讲透：
1. **一句话硬核总结** (精准提炼最核心的技术贡献)
2. **痛点与现有方案的瓶颈** (详细且专业地指出原有架构或机制到底卡在哪里)
3. **⭐ 核心创新与技术原理深度剖析** (此处为**最核心重点，篇幅必须最长**。请按步骤、模块或状态机详细展开，硬核拆解该论文是如何解决痛点的。必须把底层微架构设计、工作机制、一致性协议状态流转、数据流或算法实现清清楚楚地解释出来，绝不能只罗列结论)
4. **关键实验与数据支撑** (提炼对实际工程有指导意义的、最具代表性的性能指标提升/功耗面积开销分析)
5. **深度横评与实战启示** (客观剖析该方案的精妙处以及潜在的妥协/短板/实现代价，并谈谈如果工业界落地可能遇到的挑战)

要求：不要做浅尝辄止的表面文章，务必深入到技术细节；不要输出无关的免责声明。
如果某段分析与候选配图明显对应，请在相关段落结尾插入对应引用标记，例如 [图1]、[图2]。只能引用给出的候选配图，不要杜撰新的图号；如果没有合适的配图，就不要强行插图。"""

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

    def format_figures_summary(self, figures: Optional[List[Dict[str, Any]]]) -> str:
        """Format extracted figure metadata for the synthesis prompt."""
        if not figures:
            return "无可用候选配图。"

        lines = []
        for figure in figures:
            lines.append(
                f"- [{figure['figure_key']}] 第{figure['page']}页，图注：{figure['caption']}"
            )
        return "\n".join(lines)

    def synthesize_paper(self, paper: Paper, full_text: str, figures: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """Generate deep synthesis for a single paper."""
        if not self.is_configured():
            return None

        prompt = self.USER_PROMPT_TEMPLATE.format(
            title=paper.title,
            authors=paper.authors,
            score=paper.ai_score or 'N/A',
            tags=paper.ai_tags or 'N/A',
            abstract=paper.abstract,
            figures_summary=self.format_figures_summary(figures),
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

            figures = pdf_processor.extract_figures(paper)
            synthesis = self.synthesize_paper(paper, full_text, figures=figures)
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
