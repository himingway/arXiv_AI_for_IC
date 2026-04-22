# ArXiv 芯片架构与 EDA 前沿论文全量追踪与 AI 智能精选系统

自动追踪 ArXiv 上芯片架构、总线一致性、AI+EDA 领域的最新论文，使用 LLM 智能评分筛选，支持交互式浏览和生成深度技术推文。

## 功能特性

- **定时自动同步**: 默认每日北京时间 08:00 自动抓取 `cs.AR`, `cs.DC`, `cs.ET`, `cs.AI` 分类论文
- **AI 智能评分**: LLM 扮演资深 SoC 架构师，按相关性给出 1-10 分评分，生成推荐理由和技术标签
- **交互式看板**: Streamlit Web UI，支持按今日新增、高分、收藏筛选
- **深度推文生成**: 自动下载 PDF，提取核心章节，生成带专家点评的深度技术推文

## 领域筛选重点

1. **CPU/AI 芯片架构**: 微架构创新、存储层级优化、张量单元设计
2. **总线与一致性**: AMBA CHI/ACE 协议、Cache Coherency、NoC 拓扑
3. **EDA + AI**: 机器学习在 RTL 生成、P&R、形式化验证中的应用

## 快速开始

### 1. 安装依赖

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置 API

```bash
cp .env.example .env
# 编辑 .env，填入你的 BASE_URL 和 API_KEY
# 适配 OpenAI / 通义千问 / 智谱清言 / DeepSeek 等
```

示例配置（通义千问）：
```
BASE_URL=https://dashscope.aliyun.com/compatibility/v1
API_KEY=your_dashscope_api_key
LLM_MODEL=qwen-plus
```

### 3. 首次同步

```bash
python main.py sync
```

### 4. AI 评分

```bash
python main.py process
```

### 5. 启动交互式看板

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

访问 `http://your-server:8501` 即可使用。

## 后台常驻运行（tmux）

```bash
# 新建会话
tmux new -s arxiv

# 启动每日调度（会自动同步并后台运行）
python main.py scheduler

# Ctrl+B D 脱离会话，保持后台运行
```

## 命令行使用

| 命令 | 说明 |
|------|------|
| `python main.py sync` | 手动触发一次同步 |
| `python main.py process` | 处理所有未评分论文 |
| `python main.py scheduler` | 启动每日定时调度 |
| `python main.py stats` | 查看数据库统计信息 |

## 项目结构

```
arxiv/
├── app.py                 # Streamlit 交互式看板
├── main.py                # CLI 入口
├── requirements.txt       # Python 依赖
├── .env.example           # 环境变量示例
├── src/
│   ├── database.py        # SQLite 数据库封装
│   ├── ingest.py          # ArXiv 数据抓取
│   ├── ai_filter.py       # AI 评分筛选
│   ├── pdf_parser.py      # PDF 下载与文本提取
│   ├── deep_synthesis.py  # 深度推文生成
│   └── scheduler.py       # 定时任务调度
├── data/                  # SQLite 数据库存储
└── pdfs/                  # 下载的 PDF 缓存
```

## 数据库 Schema

- `papers`: 存储论文元数据、AI 评分、推荐理由、收藏标记
- `sync_log`: 同步日志记录

## 技术栈

- Python 3.10+
- Streamlit - Web UI
- SQLite - 本地存储
- arxiv - ArXiv API 客户端
- PyMuPDF - PDF 文本提取
- OpenAI SDK - 大模型接口（兼容所有 OpenAI 格式 API）
- APScheduler - 定时任务

## 截图

- 左侧边栏显示统计信息，提供同步和 AI 处理按钮
- 主区域按 AI 分数降序排列，支持筛选今日新增、高分必读、已收藏
- 勾选感兴趣论文后点击「生成深度推文」即可获得带专家点评的 Markdown

## License

MIT
