"""
Deep Article Synthesis module.
Generates in-depth technical blog posts/tweets from selected papers.
Supports both OpenAI-compatible APIs and Anthropic Claude API.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
import anthropic

from .database import Paper, Database
from .pdf_parser import PDFProcessor


load_dotenv()


class DeepSynthesizer:
    """Generates deep technical analysis articles from selected papers.
    Supports both OpenAI-compatible APIs and Anthropic Claude API.
    """

    SYSTEM_PROMPT = """你是一位资深的 SoC 架构专家，曾在 ARM/Intel/英伟达参与过多款商用 CPU/GPU/NPU 芯片设计。现在你需要阅读一篇最新 ArXiv 论文，并撰写一篇面向中文芯片架构工程师社区的深度技术推文。

要求：
1. 深度解析：必须深入分析论文提出的硬件机制（状态机设计、死锁避免、一致性协议状态转换、微架构流水线等）。
2. 架构师点评：专门独立一节【架构师点评】，结合工业界现状（如 ARM CMN 一致性网络、RISC-V 标准、AMD Infinity 架构等）点评：
   - 创新点的实际价值
   - RTL 实现复杂度评估
   - 对 PPA（性能、功耗、面积）的影响
   - 流片落地可行性分析
   - 值得国内团队借鉴的地方
3. 结构清晰：
   - 开头一句话总结核心贡献
   - 问题背景：这篇论文解决了什么痛点问题？
   - 核心方案：详细解释关键技术创新
   - 主要结果：论文给出了什么样的实验数据？
   - 架构师点评：你的专家见解（这部分最重要）
4. 语言风格：专业但易懂，面向有实际工程经验的芯片设计工程师，不是给本科生的科普。避免空话套话。
5. 格式：使用 Markdown 格式，不要使用英文标题，全部使用中文标题。"""

    USER_PROMPT_TEMPLATE = """论文信息：
标题：{title}
作者：{authors}
AI评分：{score}/10
标签：{tags}
摘要：{abstract}

论文核心章节文本：
---
{key_sections}
---

请按照要求生成深度技术推文："""

    def __init__(self):
        self.provider = os.getenv('LLM_PROVIDER', 'openai').lower()
        self.api_key = os.getenv('API_KEY', '')
        self.model = os.getenv('LLM_MODEL', 'gpt-4o')
        self.temperature = float(os.getenv('TEMPERATURE', '0.3'))
        self.max_tokens = int(os.getenv('MAX_TOKENS_SYNTHESIS', '16384'))

        if self.provider == 'openai':
            self.base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=180.0
            )
        elif self.provider == 'anthropic':
            self.base_url = os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')
            self.client = anthropic.Anthropic(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=180.0
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}. Use 'openai' or 'anthropic'.")

    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key and self.api_key != 'your_api_key_here')

    def synthesize_paper(self, paper: Paper, key_sections: str) -> Optional[str]:
        """Generate deep synthesis for a single paper."""
        if not self.is_configured():
            return None

        prompt = self.USER_PROMPT_TEMPLATE.format(
            title=paper.title,
            authors=paper.authors,
            score=paper.ai_score or 'N/A',
            tags=paper.ai_tags or 'N/A',
            abstract=paper.abstract,
            key_sections=key_sections
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
        result += f"生成时间：{os.getenv('DATE', '')}\n\n"
        result += "---\n\n"

        for i, paper in enumerate(papers, 1):
            print(f"Processing paper {i}/{len(papers)}: {paper.title[:50]}...")

            # Get key sections from PDF
            key_sections = pdf_processor.get_or_download(paper)
            if key_sections is None:
                result += f"## {i}. {paper.title}\n\n"
                result += f"**作者**: {paper.authors}\n\n"
                result += f"评分: **{paper.ai_score}/10** 标签: {paper.ai_tags}\n\n"
                result += f"PDF 下载失败，无法生成深度分析。\n\n"
                result += f"原文链接: {paper.pdf_url}\n\n"
                result += "---\n\n"
                continue

            synthesis = self.synthesize_paper(paper, key_sections)
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
