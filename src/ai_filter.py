"""
AI Filtering module for paper scoring and tagging.
Uses LLM to evaluate papers based on chip architecture and EDA relevance.
Supports both OpenAI-compatible APIs and Anthropic Claude API.
"""

import os
import json
import time
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import anthropic

from .database import Database, Paper


# Load environment variables from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / '.env')


class AIFilter:
    """AI-based paper filter and scorer.
    Supports both OpenAI-compatible APIs and Anthropic Claude API.
    """

    SYSTEM_PROMPT = """你是一位资深的 SoC 互联架构专家。你需要评估 ArXiv 论文与"SoC 芯片互联架构"领域的相关性，并给出 1-10 分的评分：

评分标准（越高表示越相关越重要）：
10 分：突破性创新，直接针对 Cache 一致性协议、NoC 互连网络、片上总线，具有重大理论或实践价值。必须精读。
9 分：非常相关，高质量研究，在互联架构领域有重要创新，强烈推荐阅读。
8 分：相关且有新意，技术扎实，对互联领域有参考价值，值得关注。
7 分：比较相关，主要内容围绕互联架构，有一定技术价值，可以一读。
6 分：部分内容涉及互联架构，其他部分是系统级或 core 级讨论。
4-5 分：边缘相关，只有少量内容提到互联，主要讨论其他主题。
1-3 分：不相关，论文主题是 CPU core 设计、AI 张量单元、纯算法理论、纯软件、机器学习理论不涉及硬件互联。

重点关注领域（按优先级排序）：
✅ 最高优先级：总线与一致性 — AMBA CHI/ACE 协议、Cache Coherency 机制、NoC 拓扑设计、片上互连网络、内存一致性模型、互联流量优化、一致性协议优化
✅ 次高优先级：SoC 整体架构 — 存储层级优化、互连架构创新、多芯片互联、存算一体互联
✅ 中等优先级：EDA + AI — 机器学习在物理设计、互联布线、一致性验证中的应用
❌ 低优先级（降分处理）：CPU 核设计、AI 张量单元、指令集创新、乱序执行、分支预测 — 即使有创新，也只给低分，因为不是目标领域

请严格按照 JSON 格式输出，不要其他文字：
{
  "score": 1-10 的整数,
  "reason": "100字以内的推荐理由，说明为什么给这个分，重点说创新点在哪里",
  "tags": "逗号分隔的技术标签，例如 一致性协议, NoC, CHI, ACE, 片上互联, 缓存一致性 等"
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
        self.timeout = float(os.getenv('TIMEOUT_SCORING', '120'))  # 2 minutes timeout for scoring

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

        # Fetch only as many papers as needed to avoid full table scan on each call
        unprocessed = db.get_unprocessed_papers(limit=batch_size)
        processed = 0

        for i, paper in enumerate(unprocessed):
            print(f"Processing {i+1}/{len(unprocessed)}: {paper.title[:60]}...")
            score, reason, tags = self.analyze_paper(paper)

            if score is not None:
                db.update_ai_analysis(paper.id, score, reason, tags)
                processed += 1

            if i < len(unprocessed) - 1 and delay_seconds > 0:
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
