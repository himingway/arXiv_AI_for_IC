# ArXiv 芯片架构与 EDA 前沿论文全量追踪与 AI 智能精选系统

自动追踪 ArXiv 上芯片架构、总线一致性、AI+EDA 领域的最新论文，使用 LLM 智能评分筛选，支持交互式浏览和生成深度技术推文。

## 功能特性

- **定时自动同步**: 默认每日北京时间 08:00 自动抓取 `cs.AR`, `cs.DC`, `cs.ET`, `cs.AI` 分类论文
- **ArXiv 速率限制合规**: `arxiv.Client(delay_seconds=3)` 在分页请求间自动限速，单连接顺序抓取
- **增量同步优化**: 滑动窗口提前停止策略，遇到连续已存在论文自动停止
- **支持双 LLM 协议**: 统一客户端工厂，兼容 OpenAI 协议和原生 Anthropic 协议
- **AI 智能评分**: LLM 扮演资深 SoC 架构师，按相关性给出 1-10 分评分，生成推荐理由和技术标签
- **交互式看板**: Streamlit Web UI，支持分页浏览、按今日新增/高分/收藏筛选
- **启动即检查今日同步状态**: 侧边栏实时提示同步状态，支持一键补跑
- **深度推文生成**: 自动下载 PDF 提取核心章节，逐篇生成 Markdown，支持断点续传和手动取消
- **结构化日志**: 统一 Python logging 框架，后台运行日志可分级过滤
- **数据库优化**: WAL 模式 + 5 个索引 + 线程本地连接，支持并发读写

## 领域筛选重点（当前配置侧重互联架构）

1. **⭐ 最高优先级：总线与一致性** - AMBA CHI/ACE 协议、Cache Coherency 机制、NoC 拓扑设计、片上互连网络、内存一致性模型、互联流量优化、一致性协议优化
2. **⭐ 次高优先级：SoC 整体架构** - 存储层级优化、多芯片互联、存算一体互联架构
3. **⭐ 中等优先级：EDA + AI** - 机器学习在 RTL 生成、物理设计（Placement & Routing）、互联布线、形式化验证中的应用
4. **⚠️ 低优先级（自动降分）** - CPU 核心设计、AI 张量单元、指令集创新

## 支持的 LLM 提供商

| 提供商 | 协议 | 配置示例 |
|--------|------|----------|
| OpenAI 官方 | OpenAI | `LLM_PROVIDER=openai`, `BASE_URL=https://api.openai.com/v1` |
| 通义千问 | OpenAI 兼容 | `LLM_PROVIDER=openai`, `BASE_URL=https://dashscope.aliyun.com/compatibility/v1` |
| DeepSeek | OpenAI 兼容 | `LLM_PROVIDER=openai`, `BASE_URL=https://api.deepseek.com/v1` |
| 智谱 GLM | OpenAI 兼容 | `LLM_PROVIDER=openai`, `BASE_URL=https://open.bigmodel.cn/api/paas/v4` |
| 字节豆包 | OpenAI 兼容 | `LLM_PROVIDER=openai`, `BASE_URL=https://ark.cn-beijing.volces.com/api/v3` |
| 火山方舟 | OpenAI 兼容 | `LLM_PROVIDER=openai`, `BASE_URL=https://ark.cn-beijing.volces.com/api/v3` |
| Anthropic Claude | Anthropic 原生 | `LLM_PROVIDER=anthropic`, `ANTHROPIC_BASE_URL=https://api.anthropic.com` |

## 快速开始

### 1. 安装依赖

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

或使用 uv:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. 配置 API

```bash
cp .env.example .env
# 编辑 .env，填入你的配置
```

**示例配置 - 火山方舟 OpenAI 兼容（推荐国内用户）:**
```env
LLM_PROVIDER=openai
BASE_URL=https://ark.cn-beijing.volces.com/api/v3
API_KEY=your-ark-api-key
LLM_MODEL=ep-xxxxxxxxxxxxxxxx  # 你的端点 ID
TEMPERATURE=0.3
MAX_TOKENS_SYNTHESIS=16384
ARXIV_CATEGORIES=cs.AR,cs.DC,cs.ET,cs.AI
TIMEOUT_SCORING=120
TIMEOUT_SYNTHESIS=300
TIMEOUT_DOWNLOAD=120
DB_PATH=./data/papers.db
PDF_DIR=./pdfs
```

**示例配置 - Anthropic Claude:**
```env
LLM_PROVIDER=anthropic
ANTHROPIC_BASE_URL=https://api.anthropic.com
API_KEY=your-anthropic-key
LLM_MODEL=claude-3-5-sonnet-20241022
TEMPERATURE=0.3
MAX_TOKENS_SYNTHESIS=16384
DB_PATH=./data/papers.db
PDF_DIR=./pdfs
```

### 3. 一键启动服务

```bash
./start.sh
```

同时启动 Web 看板和每日调度器，访问 `http://localhost:8501`。

```bash
./stop.sh       # 停止服务
./status.sh     # 查看状态
```

### 4. 命令行使用

| 命令 | 说明 |
|------|------|
| `python main.py sync` | 手动同步，默认抓取最新 100 篇 |
| `python main.py sync N` | 指定抓取 N 篇，例如 `python main.py sync 500` |
| `python main.py process` | 处理所有未评分论文 |
| `python main.py scheduler` | 启动每日定时调度 |
| `python main.py stats` | 查看数据库统计信息 |
| `python main.py debug` | 调试：打印当前 LLM 配置 |
| `python main.py clear` | 清空数据库（需要确认） |
| `python main.py clear --pdf` | 清空数据库 AND 删除所有 PDF |

## 项目结构

```
├── app.py                 # Streamlit 交互式看板（分页、断点续传、取消生成）
├── main.py                # CLI 入口
├── start.sh / stop.sh / status.sh  # 服务管理脚本
├── requirements.txt       # Python 依赖
├── .env.example           # 环境变量配置模板
└── src/
    ├── __init__.py
    ├── database.py        # SQLite 数据库（WAL 模式、5 个索引、线程本地连接）
    ├── ingest.py          # ArXiv 数据抓取（合规限速 + 提前停止优化）
    ├── ai_filter.py       # AI 评分筛选
    ├── deep_synthesis.py  # 深度推文生成
    ├── pdf_parser.py      # PDF 下载（URL 安全校验）与文本提取
    ├── scheduler.py       # 每日定时调度
    ├── llm_client.py      # 共享 LLM 客户端工厂（OpenAI / Anthropic）
    └── logging_config.py  # 统一日志配置
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LLM_PROVIDER` | `openai` | LLM 协议：`openai` 或 `anthropic` |
| `API_KEY` | — | API 密钥 |
| `LLM_MODEL` | `gpt-4o` | 模型名称 |
| `TEMPERATURE` | `0.3` | 生成温度 0.0-1.0 |
| `ARXIV_CATEGORIES` | `cs.AR,cs.DC,cs.ET,cs.AI` | 抓取分类，逗号分隔 |
| `MAX_RESULTS` | `100` | 每次同步最大抓取数量 |
| `TIMEOUT_SCORING` | `120` | AI 评分超时（秒） |
| `TIMEOUT_SYNTHESIS` | `300` | 深度推文生成超时（秒） |
| `TIMEOUT_DOWNLOAD` | `120` | PDF 下载超时（秒） |
| `MAX_TOKENS_SCORING` | `2000` | AI 评分最大输出长度 |
| `MAX_TOKENS_SYNTHESIS` | `16384` | 深度推文最大输出长度 |
| `DB_PATH` | `./data/papers.db` | 数据库路径 |
| `PDF_DIR` | `./pdfs` | PDF 存储目录 |

## 同步说明

### ArXiv API 合规性

本项目严格遵守 [ArXiv API 官方速率限制](https://arxiv.org/help/api):
- **单连接请求**: 不使用多连接并发抓取
- **分页自动限速**: `arxiv.Client(delay_seconds=3)` 在分页请求间自动添加 3 秒间隔
- **稳定可靠**: 即使同步上千篇也不会被限流封杀

### 提前停止策略（鲁棒滑动窗口）

ArXiv 按提交时间降序排序（新 → 旧），使用滑动窗口检测最近 20 篇论文状态：
- **只有抓到至少一篇新论文后**，才启用提前停止判断
- 只有当**最近 20 篇全部都已经在数据库中**时，才会真正提前停止
- 如果本次同步从头到尾一篇新论文都没抓到，会遍历完所有 `max_results` 才结束

## Web 看板功能

- 分页浏览论文（每页 50 篇），支持上/下页和页码跳转
- 筛选：仅今日新增 / 仅高分必读（≥7 分）/ 仅已收藏 / 关键词搜索
- 勾选论文后点击「生成深度推文」，自动下载 PDF 提取核心章节，逐篇生成 Markdown
- 生成过程支持**断点续传**（刷新后自动恢复）和**手动取消**
- 生成后可直接下载单篇或全部 Markdown 文件

## 深度推文内容结构

1. **一句话硬核总结** (精准提炼最核心的技术贡献)
2. **痛点与现有方案的瓶颈** (详细且专业地指出原有架构或机制到底卡在哪里)
3. **⭐ 核心创新与技术原理深度剖析** (最核心部分：按步骤、模块硬核拆解)
4. **关键实验与数据支撑** (性能指标提升/功耗面积开销分析)
5. **深度横评与实战启示** (精妙处与潜在短板/代价评估，工业界落地挑战)

## 数据库优化

| 优化项 | 说明 |
|--------|------|
| **WAL 模式** | 支持并发读写，Web 看板与调度器同时运行不冲突 |
| **5 个索引** | `ai_processed`、`ai_score`、`created_at`、`is_starred`、`published` |
| **线程本地连接** | 每个线程复用同一连接，减少连接开销 |
| **busy_timeout** | 5 秒等待锁，避免并发写入时立即报错 |

## 技术栈

- Python 3.10+
- Streamlit — Web UI
- SQLite (WAL) — 本地存储
- arxiv — ArXiv API 客户端
- PyMuPDF (fitz) — PDF 文本提取
- OpenAI SDK / Anthropic SDK — 双协议 LLM 支持
- APScheduler — 定时任务调度

## License

MIT
