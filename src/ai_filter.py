"""
AI Filtering module for paper scoring and tagging.
Uses LLM to evaluate papers based on chip architecture and EDA relevance.
Supports both OpenAI-compatible APIs and Anthropic Claude API.
"""

import os
import json
import time
from typing import Tuple, Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
import anthropic

from .database import Database, Paper


# Load environment variables
load_dotenv()


class AIFilter:
    """AI-based paper filter and scorer.
    Supports both OpenAI-compatible APIs and Anthropic Claude API.
    """

    SYSTEM_PROMPT = """你是一位资深的 SoC 架构师与 EDA 领域专家。你需要评估 ArXiv 论文与"芯片架构与 EDA 前沿"领域的相关性，并给出 1-10 分的评分：

评分标准（越高表示越相关越重要）：
10 分：突破性创新，直接针对 CPU/AI 芯片微架构、Cache 一致性协议、NoC、或 AI 驱动的 EDA 方法，具有重大理论或实践价值。必须精读。
9 分：非常相关，高质量研究，对工业界或学术界有重要参考价值。强烈推荐阅读。
8 分：相关且有新意，技术扎实，值得领域内关注。
7 分：比较相关，有一定技术价值，可以一读。
6 分：部分相关，某些概念或方法可能有参考意义。
4-5 分：边缘相关，只有少量内容涉及目标领域。
1-3 分：不相关，属于其他领域（如纯算法理论、纯软件、纯机器学习理论不涉及硬件）。

重点关注领域：
1. CPU/AI 芯片架构：微架构创新、存储层级优化、张量单元设计、指令集创新、乱序执行、分支预测。
2. 总线与一致性：AMBA CHI/ACE 协议、Cache Coherency 机制、NoC 拓扑、互连网络、内存一致性模型。
3. EDA + AI：机器学习在 RTL 生成、物理设计（Placement & Routing）、逻辑综合、形式化验证、时序分析中的应用。

请严格按照 JSON 格式输出，不要其他文字：
{
  "score": 1-10 的整数,
  "reason": "100字以内的推荐理由，说明为什么给这个分，重点说创新点在哪里",
  "tags": "逗号分隔的技术标签，例如 CPU架构, 一致性协议, NoC, AI EDA, RTL生成 等"
}"""

    USER_PROMPT_TEMPLATE = """论文标题：{title}
作者：{authors}
分类：{categories}
摘要：{abstract}

请按照要求给出评分、理由和标签："""

    def __init__(self):
        self.provider = os.getenv('LLM_PROVIDER', 'openai').lower()
        self.api_key = os.getenv('API_KEY', '')
        self.model = os.getenv('LLM_MODEL', 'gpt-4o')
        self.temperature = float(os.getenv('TEMPERATURE', '0.1'))
        self.max_tokens = int(os.getenv('MAX_TOKENS_SCORING', '2000'))

        if self.provider == 'openai':
            self.base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=60.0
            )
        elif self.provider == 'anthropic':
            self.base_url = os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')
            self.client = anthropic.Anthropic(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=60.0
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}. Use 'openai' or 'anthropic'.")

    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key and self.api_key != 'your_api_key_here')

    def analyze_paper(self, paper: Paper) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """Analyze a paper and return score, reason, and tags."""
        prompt = self.USER_PROMPT_TEMPLATE.format(
            title=paper.title,
            authors=paper.authors,
            categories=paper.categories,
            abstract=paper.abstract
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
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"}
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
                    max_tokens=self.max_tokens,
                )
                content = response.content[0].text.strip()

            result = json.loads(content)

            score = int(result.get('score', 1))
            reason = result.get('reason', '')
            tags = result.get('tags', '')

            # Clamp score to 1-10
            score = max(1, min(10, score))

            return score, reason, tags

        except Exception as e:
            print(f"Error analyzing paper {paper.id}: {e}")
            return None, None, None

    def process_next_batch(self, db: Database, batch_size: int = 10, delay_seconds: float = 2.0) -> int:
        """Process next batch of unprocessed papers. Returns number processed successfully."""
        if not self.is_configured():
            print("API key not configured. Cannot process papers.")
            return 0

        unprocessed = db.get_unprocessed_papers()
        processed = 0

        for i, paper in enumerate(unprocessed[:batch_size]):
            print(f"Processing {i+1}/{min(batch_size, len(unprocessed))}: {paper.title[:60]}...")
            score, reason, tags = self.analyze_paper(paper)

            if score is not None:
                db.update_ai_analysis(paper.id, score, reason, tags)
                processed += 1

            if i < batch_size - 1 and delay_seconds > 0:
                time.sleep(delay_seconds)

        return processed

    def get_processing_stats(self, db: Database) -> Dict[str, Any]:
        """Get AI processing statistics."""
        stats = db.get_stats()
        return {
            'total': stats['total_papers'],
            'processed': stats['processed_papers'],
            'pending': stats['unprocessed_papers'],
            'average_score': stats['average_score']
        }


def process_all_unprocessed(db: Database, batch_size: int = 10, delay: float = 2.0) -> int:
    """Process all unprocessed papers in batches."""
    ai_filter = AIFilter()
    total_processed = 0

    while True:
        pending = db.count_unprocessed()
        if pending == 0:
            break

        print(f"Processing batch, {pending} papers remaining...")
        processed = ai_filter.process_next_batch(db, batch_size=min(batch_size, pending), delay_seconds=delay)
        total_processed += processed

        if processed == 0:
            # No progress made, likely API issue
            break

    return total_processed
