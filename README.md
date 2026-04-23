# ArXiv 芯片架构与 EDA 前沿论文全量追踪与 AI 智能精选系统

自动追踪 ArXiv 上芯片架构、总线一致性、AI+EDA 领域的最新论文，使用 LLM 智能评分筛选，支持交互式浏览和生成深度技术推文。

## 功能特性

- **定时自动同步**: 默认每日北京时间 08:00 自动抓取 `cs.AR`, `cs.DC`, `cs.ET`, `cs.AI` 分类论文
- **ArXiv 速率限制合规**: 严格遵守官方要求 - 单连接、每请求间隔 3 秒，稳定抓取不会被限流
- **增量同步优化**: 提前停止策略，遇到连续已存在论文自动停止，说明所有更新论文已入库，避免浪费时间
- **支持双 LLM 协议**: 兼容 OpenAI 协议（OpenAI/Qwen/GLM/DeepSeek/Kimi/Doubao/火山方舟）和原生 Anthropic 协议（Claude）
- **AI 智能评分**: LLM 扮演资深 SoC 架构师，按相关性给出 1-10 分评分，生成推荐理由和技术标签
- **交互式看板**: Streamlit Web UI，支持按今日新增、高分、收藏筛选
- **启动即检查今日同步状态**: 看板侧边栏实时提示“今日已同步/未同步”，并支持一键“补跑今日同步”
- **深度推文生成**: 自动下载 PDF，提取核心章节，生成带专家点评的深度技术推文

## 领域筛选重点（当前配置侧重互联架构）

1. **⭐ 最高优先级：总线与一致性** - AMBA CHI/ACE 协议、Cache Coherency 机制、NoC 拓扑设计、片上互连网络、内存一致性模型、互联流量优化、一致性协议优化
2. **⭐ 次高优先级：SoC 整体架构** - 存储层级优化、多芯片互联、存算一体互联架构
3. **⭐ 中等优先级：EDA + AI** - 机器学习在 RTL 生成、物理设计（Placement & Routing）、互联布线、形式化验证中的应用
4. **⚠️ 低优先级（自动降分）** - CPU 核心设计、AI 张量单元、指令集创新（即使有创新也会降低评分，因为不是目标领域）

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

**使用 pip:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**使用 uv:**
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

### 3. 首次同步

默认抓取最新 100 篇（调试快速）：
```bash
python main.py sync
```

指定抓取数量：
```bash
python main.py sync 500    # 抓取最新 500 篇
python main.py sync 1000   # 抓取最新 1000 篇
```

### 4. AI 评分

```bash
python main.py process
```

会分批处理所有未评分论文，遇到错误自动重试，中断后下次继续处理。

### 5. 启动交互式看板

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

使用 uv：
```bash
uv run streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

访问 `http://your-server-ip:8501` 即可使用。

## 后台常驻运行（tmux）

### Web 看板后台运行

```bash
tmux new -s arxiv-dashboard
source .venv/bin/activate
uv run streamlit run app.py --server.address 0.0.0.0 --server.port 8501
# 按 Ctrl+B 然后 D 退出会话，保持后台运行
```

### 每日自动同步

```bash
tmux new -s arxiv-scheduler
source .venv/bin/activate
uv run python main.py scheduler
# 按 Ctrl+B 然后 D 退出会话
```

调度器会：
- 每天 **北京时间 08:00** 自动同步最新论文（此时 ArXiv 已完成日更）
- 启动时如果今天还没同步过，会立即同步一次
- 自动同步后自动处理 10 篇新论文

## 命令行使用

| 命令 | 说明 |
|------|------|
| `python main.py sync` | 手动同步，默认抓取最新 100 篇 |
| `python main.py sync N` | 指定抓取 N 篇，例如 `python main.py sync 500` |
| `python main.py process` | 处理所有未评分论文 |
| `python main.py scheduler` | 启动每日定时调度 |
| `python main.py stats` | 查看数据库统计信息 |
| `python main.py debug` | 调试：打印当前 LLM 配置 |
| `python main.py clear` | 清空数据库（需要确认） |
| `python main.py clear --pdf` | 清空数据库 AND 删除所有下载的 PDF |

## 项目结构

```
arxiv/
├── app.py                 # Streamlit 交互式看板
├── main.py                # CLI 入口，支持所有命令
├── requirements.txt       # Python 依赖
├── .env.example           # 环境变量配置模板
├── .gitignore            # Git 忽略规则
└── src/
    ├── __init__.py
    ├── database.py        # SQLite 数据库封装（自动创建父目录）
    ├── ingest.py          # ArXiv 数据抓取（合规速率限制+提前停止优化）
    ├── ai_filter.py       # AI 评分筛选（支持双协议）
    ├── pdf_parser.py      # PDF 下载与核心章节文本提取
    ├── deep_synthesis.py  # 深度推文生成（支持双协议）
    └── scheduler.py       # 每日定时调度
```

## 生成参数配置（可在 .env 中调整）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TEMPERATURE` | 0.3 | 0.0-1.0，越低越稳定，越高越有创造性 |
| `ARXIV_CATEGORIES` | `cs.AR,cs.DC,cs.ET,cs.AI` | 抓取分类，逗号分隔 |
| `MAX_RESULTS` | 100 | 每次同步最大抓取数量（scheduler默认每次抓取最新 100 篇） |
| `TIMEOUT_SCORING` | 120 | AI评分接口超时（秒） |
| `TIMEOUT_SYNTHESIS` | 300 | 深度推文生成接口超时（秒） |
| `TIMEOUT_DOWNLOAD` | 120 | PDF 下载超时（秒） |
| `MAX_TOKENS_SCORING` | 2000 | AI评分最大输出长度 |
| `MAX_TOKENS_SYNTHESIS` | 16384 | 深度推文最大输出长度，可根据模型支持调大到 65536 |

## 同步说明

### ArXiv API 合规性

本项目严格遵守 [ArXiv API 官方速率限制](https://arxiv.org/help/api):
- **单连接请求**: 不使用多连接并发抓取
- **每 3 秒一个请求**: 严格保证间隔，避免触发 429 限流
- **稳定可靠**: 即使同步上千篇也不会被限流封杀

### 提前停止策略

因为 ArXiv 按提交时间降序排序，如果连续遇到 20 篇已存在论文，说明所有更新的论文已经入库，后面更老的肯定都已经有了，可以安全提前停止，节省大量时间。**已经发现的新论文一定会全部抓取完成，不会漏抓**。

因为 ArXiv 按提交时间降序排序，如果连续遇到 20 篇已存在论文，说明所有更新的论文已经入库，后面更老的肯定都已经有了，可以安全提前停止，节省大量时间。**已经发现的新论文一定会全部抓取完成，不会漏抓**。

## Web 看板功能

- 默认按 AI 评分降序排列（高分在前）
- 筛选选项：仅今日新增 / 仅高分必读 (≥7分) / 仅已收藏
- 搜索支持：按标题/作者/标签搜索
- 勾选论文后点击「生成深度推文」，自动下载 PDF 提取核心章节，生成带架构师点评的 Markdown
- 生成后可直接下载 Markdown 文件

## 深度推文内容结构

按以下高深度和硬核技术逻辑组织，注重解析原理解密（标题表达可灵活调整）：
1. **一句话硬核总结** (精准提炼最核心的技术贡献)
2. **痛点与现有方案的瓶颈** (详细且专业地指出原有架构或机制到底卡在哪里)
3. **⭐ 核心创新与技术原理深度剖析** (最核心部分：按步骤、模块硬核拆解，剖析底层微架构设计、工作机制、一致性协议状态流转、数据流或算法实现，把技术原理讲透)
4. **关键实验与数据支撑** (提炼最具代表性的性能指标提升/功耗面积开销分析)
5. **深度横评与实战启示** (客观剖析其精妙处与潜在的短板/代价评估，以及工业界落地的挑战)

## 数据库 Schema

- `papers`: 存储论文元数据、AI 评分、推荐理由、标签、收藏标记
- `sync_log`: 同步日志记录，包含每次同步添加数量和状态

## 技术栈

- Python 3.10+
- Streamlit - Web UI
- SQLite - 本地轻量存储
- arxiv - ArXiv API 客户端
- PyMuPDF (fitz) - PDF 文本提取
- OpenAI SDK - OpenAI 协议支持
- Anthropic SDK - Anthropic 协议支持
- APScheduler - 定时任务调度

## License

MIT
