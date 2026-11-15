[English](./README.md) | [简体中文](./README_zh.md)

# speech-sieve

**speech-sieve** 是一个面向语音数据集构建的模块化音频质检与对齐审计工具（**Apache-2.0** 协议）。输入音频清单（JSONL），即可输出逐条声学体检指标、可选的语音-文本对齐核验与串音风险报告，并将样本自动分流为 **正常（`normal`）**、**建议人工复听（`review_recommended`）** 与 **明确异常（`alert`）**。

核心信号算子仅依赖 Python 标准库、`NumPy` 与 `PyYAML`，纯 CPU 环境即可开箱体验；Silero VAD、Qwen3-ForcedAligner 与 `pyannote` 多人串音检测均作为按需加载的可选扩展提供。

---

## 为什么做这个工具（解决什么问题与适用场景）

在构建语音合成（TTS）、声音克隆、语音识别（ASR）或语音大模型训练语料时，真实采集与众包录音常面临三类典型难题：

1. **隐蔽录音瑕疵难以仅靠时长和音量发现**：录音中可能夹杂口吃卡顿与“念错后重读同一句”、旁人插话串音、未按文本朗读的多余闲聊、大段静音或首尾截断，以及问答采集中误将屏幕上的题目提示词一并念出。
2. **单一全局阈值或纯 ASR 字错率（CER）容易误伤高价值样本**：面对方言口音、多音字、情感朗读或高表现力语音时，通用 ASR 往往识别率偏低，按固定 CER 阈值一刀切会误删大量珍贵语料；同样，直接将停顿次数、语速或频谱亮度作为硬淘汰门槛，也容易把正常的语流节奏当作故障丢弃。
3. **缺少明确的状态区分，容易把“未运行检测”误当“通过”**：很多脚本在未安装某个模型或音频过短无法计算统计量时默认填 `0` 放行，导致质检结果无法真实反映覆盖情况。

**典型适用场景**：
- 有稿朗读 / 配音 / 众包采集语音数据集入库前质检与人工复听优先级排序；
- 自由口语 / 访谈切分语料的多人串音筛查、声学电平体检与题干误读排查；
- 作为纯 Python / NumPy 基础声学算子库，集成到更大的音频清洗或训练前处理流水线中。

---

## 处理流程与各步骤实现逻辑

```text
输入清单 (manifest.jsonl: audio_path, reference_text?, question?)
   │
   ├─► [1] 音频读取与基础波形检查 (audio.py, features.py)
   │        └─ 格式校验/重采样/单声道转换、空波形与削波检测、数字电平统计
   │
   ├─► [2] 多维声学画像统计 (features.py)
   │        └─ 自适应帧长帧移、多窗电平动态波动、停顿节奏、2048 点 FFT 频谱分布
   │
   ├─► [3] 无监督声学重录与口吃检测 (repeat.py)
   │        └─ 多尺度降采样 MFCC 粗检索 + 子序列动态时间规整 (DTW) 匹配句内重复
   │
   ├─► [4] 语音端点检测与强制对齐核验 [可选] (models.py, alignment.py)
   │        └─ Silero VAD + Qwen3-ForcedAligner 对齐范围比对、文本/声学重复交叉核验
   │
   ├─► [5] 多人串音检测与题干误读识别 [可选] (models.py, topic_prefix.py)
   │        └─ pyannote 跨样本滑窗批推理检测重叠说话 + 问答前缀匹配检测念题
   │
   └─► [6] 显式三态统计与分流输出 (pipeline.py, cli.py)
            └─ 长度分桶与 OOM 批减半保护 -> 输出 report.jsonl (normal / review_recommended / alert)
```

### 各步骤核心方法设计

#### 步骤 1：跨平台音频读取与基础波形检查（`audio.py`、`features.py`）
- **多后端容错解码**：依次尝试 `soundfile` → Python 标准库 `wave` → `scipy.io.wavfile` 读取音频，无重度第三方解码库也能稳定处理标准 WAV。
- **可配的格式策略**：在开发或兼容多种来源语料时，可开启自动转单声道与线性重采样至目标采样率（如 `16 kHz`）；在需要严格验收录制规格时，可通过配置关闭自动转换直接拦截非标音频。
- **硬异常拦截**：实时检测零长/全零静音文件以及满刻度削波（Clipping）占比，防止损坏文件流入下游训练。

#### 步骤 2：自适应声学动态、节奏与频谱画像（`features.py`）
- **采样率自适应与 Nyquist 安全频带**：所有短时帧长与帧移根据输入采样率实时换算；频谱能量带严格受限于当前采样率的 Nyquist 上限（兼容 `8 kHz` 电话窄带与 `16 kHz+` 宽带录音），高采样频带在窄带下自动标记未评估而非输出失真零值。
- **区分“硬故障”与“连续描述特征”**：
  - 统计有效语音段的数字信号幅度（`speech_rms_dbfs`）、长句多窗电平动态范围（`P90 - P10`）与不同能量深度下的短停顿分布；
  - 采用高分辨率 **`nfft = 2048`** FFT 提取频谱 95% 滚降频率、高低频带占比及平坦度，减少粗频率步长在低频处带来的量化误差；
  - 默认策略下，语速、停顿节奏与频谱倾斜仅作为连续画像指标随报告输出（用于整体分布监控或人工抽样），不直接作为自动拒收依据；且停顿率等依赖足够时长统计的指标在短句不足统计底限时不会强行外推告警。

#### 步骤 3：不依赖 ASR 的声学重录与口吃检测（`repeat.py`）
- **解决痛点**：录音员卡壳、口吃或读错半句紧接着重读时，转写文本里往往只有正确的那一句，单靠文本无法发现音频中多出来的重复段落。
- **实现方法**：直接在波形上抽取梅尔倒谱特征（MFCC），按多尺度窗口长度（默认 `0.48 s`、`0.96 s`、`1.92 s`）对活跃语音帧建立降采样粗描述子并做分块距离检索，筛选高相似度候选片段对后，运行**子序列动态时间规整（Subsequence DTW）**。
- **实用性保障**：DTW 路径支持 `0.67×–1.5×` 的时长伸缩比例，即使前后两次朗读语速不同也能准确检出；同时基于帧移时长动态推算时间戳，并设有单文件耗时预算上限，避免超长音频拖慢批处理吞吐。

#### 步骤 4：VAD 与强制对齐交叉核验（`models.py`、`alignment.py`，可选）
- **语音与文本范围比对**：结合 Silero VAD 与 Qwen3-ForcedAligner 对齐结果，对比“VAD 检出人声总区间”与“参考文本实际对齐的时间包络”，定量识别文本前后夹杂的多余说话声、过长首尾空白或静音占主导的录音。
- **声学重复 × 文本语义交叉放行**：当步骤 3 检测到音频中有两段声学高度相似时，结合步骤 4 的字符/词级对齐区间核对参考文本——如果这两段声音恰好各自落在参考文本本来就包含叠词或重复语句的位置，则说明属于**合法文本复现**并自动豁免，仅把“在对齐包络之外存在多余有效发音”的异常重录送入告警或复核队列。
- **跨语种词元估算**：支持按空格分词（英语等）、按字符（中文/日文等）以及自动混合切分，使停顿与语速统计适用于多语种语料。

#### 步骤 5：多人串音与题干误读插件（`models.py`、`topic_prefix.py`，可选）
- **跨样本滑窗批推理检测多人串音（`BatchedOverlapAdapter`）**：封装 `pyannote/segmentation-3.0`，将整批音频的滑动分析窗跨文件拼接打包执行向量化前向推理，大幅减少零散短音频逐条调用的模型启动开销；同时针对短于 `10 s` 滑窗长度的片段保留右侧补零区间的有效插值权重标记，防止短句串音风险被静默忽略。
- **问答前缀 / 题干误读识别（`topic_prefix.py`）**：面向问答、自由口语等带有 `question` 字段的采集场景，通过正则前缀剥离、语气停顿词跳过及 `SequenceMatcher` 首句模糊重合度匹配，识别说话人开头误读提示词或题干的录音；所有自定义引导模板均外置在 YAML 中配置。

#### 步骤 6：显式三态报告与稳健批调度（`pipeline.py`、`cli.py`）
- **显式状态机**：每一项检测结果在输出 JSONL 中都清晰标注为三态之一：
  - `evaluated`：满足计算前提并已真实完成评估；
  - `not_evaluated`：所需可选模型未开启或样本长度/采样率未达到该指标的有效统计条件（附具体原因）；
  - `skipped`：当前运行模式或配置显式关闭（例如在 `spontaneous` 自由口语模式下自动跳过依赖参考稿的对齐覆盖检测）。
- **批处理稳定性**：对齐与重模型批处理内置**按音频长度分桶排序**减少 Padding 浪费，并在遭遇 GPU 显存不足（OOM）时**自动将当前 Batch 规模减半递归重试**，精确定位个别极长或异常样本而不中断整批任务。

---

## 安装与快速开始

### 1. 安装

最小安装（纯 CPU 环境，仅依赖 Python ≥ 3.10、NumPy、PyYAML）：

```bash
pip install -e .
```

按需安装可选音频解码或深度学习模型扩展：

```bash
pip install -e ".[soundfile]"   # 扩充多种音频格式读取（未安装时原生支持标准 WAV）
pip install -e ".[vad]"         # 启用 Silero VAD 语音端点检测 (MIT)
pip install -e ".[align]"       # 启用 Qwen3-ForcedAligner-0.6B 强制对齐 (Apache-2.0)
pip install -e ".[overlap]"     # 启用 pyannote.audio 多人重叠说话检测 (MIT，需 HF 授权)
```

> **使用 `[overlap]` 说明**：`pyannote/segmentation-3.0` 为 Hugging Face 门控受限仓库。启用前请先在 Hugging Face 模型主页同意访问条款并设置本机 `HF_TOKEN`，或将配置指向本地已授权下载的模型权重路径。详见 [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md)。

### 2. 三步跑通合成样例（无需真实语料与 GPU）

以下命令完全在合成 WAV 数据上运行（在 Windows PowerShell 下先运行 `$env:PYTHONPATH="src"`）：

```bash
# 1. 生成 5 条包含正常、削波、题干误读、句内重录、窄带双声道的合成测试音频与清单
PYTHONPATH=src python examples/generate_demo.py --out-dir demo_data

# 2. 校验 YAML 配置文件合法性
PYTHONPATH=src python -m speech_sieve.cli check --config configs/default.yaml

# 3. 使用默认中性配置运行纯 CPU 质检并生成报告
PYTHONPATH=src python -m speech_sieve.cli run \
  --input demo_data/manifest.jsonl \
  --config configs/default.yaml \
  --output demo_data/report.jsonl
```

运行全套标准库 `unittest` 验证：

```bash
PYTHONPATH=src python -B -m unittest discover -s tests -v
```

---

## 配置与场景适配指南

仓库提供唯一的中性默认配置文件 [`configs/default.yaml`](./configs/default.yaml)（内置完整注释，纯 CPU 默认开关即可开箱运行）。你可以直接复制一份该文件，根据自身数据形态按需调整顶层开关：

| 常见业务场景 | 建议调整的核心配置项 | 效果说明 |
| --- | --- | --- |
| **纯 CPU 轻量筛查** | 保持 `configs/default.yaml` 默认值不变 | 零额外模型依赖，完成格式校验、数字电平、削波、句内声学重复与题干首句模糊比对 |
| **有稿朗读 / 配音 / TTS 建库** | 保持 `mode: scripted_reading`，开启 `models.vad.enabled: true`、`models.aligner.enabled: true`、`alignment.enabled: true`（可选开启 `overlap`）；若需严格卡录制规格则设 `audio.require_mono: true` 且关闭 `allow_resample`/`allow_downmix` | 解锁文本强制对齐覆盖率、静音比、声学重复与对齐文本交叉豁免 |
| **自由口语 / 问答 / 访谈** | 设 `mode: spontaneous`；开启 `vad` 与 `overlap`；在 `topic_prefix` 中填入自定义常见题干前缀、口头禅填充词 | 自动跳过强文本对齐约束，专注多人串音与开头误读题干筛查 |

---

## 仓库目录结构与使用边界

```text
.
├── pyproject.toml                  # 包配置 (speech-sieve / speech_sieve) 与可选扩展组
├── LICENSE                         # Apache License 2.0
├── THIRD_PARTY_NOTICES.md          # 第三方依赖许可与门控模型授权指引
├── README.md                       # 英文文档
├── README_zh.md                    # 简体中文文档
├── configs/
│   └── default.yaml                # 唯一中性默认配置（附全量字段注释）
├── src/speech_sieve/
│   ├── __init__.py
│   ├── audio.py                    # 跨平台音频读取、单声道化与重采样
│   ├── config.py                   # YAML 配置加载、物理参数校验与跨平台原子写
│   ├── features.py                 # 波形完整性、电平动态、停顿节奏、FFT 频谱统计
│   ├── repeat.py                   # 多尺度 MFCC + 子序列 DTW 声学自重复检测
│   ├── alignment.py                # 语种自适应分词、对齐范围覆盖率与重复交叉核验
│   ├── topic_prefix.py             # 可配置题干误读与开头提示词检测
│   ├── models.py                   # VAD / ForcedAligner / BatchedOverlap 推理适配器与 OOM 保护
│   ├── pipeline.py                 # 检查编排、三态状态机记录与样本分流路由
│   └── cli.py                      # check / run 命令行入口
├── examples/
│   └── generate_demo.py            # 合成 WAV 演示数据生成器
└── tests/
    ├── test_operators.py
    ├── test_gating_and_config.py
    └── test_mock_pipeline.py
```

**已知使用边界**：
- 数字信号电平（`dBFS`）衡量波形数字幅度，不等同于物理声压级或主观听感清晰度（MOS/PESQ）。
- `overlap_risk`、`prompt_echo_prefix` 与可选 `advisory_cer` 均定位为人工抽听排序辅助信号，不建议脱离人工复核直接作为自动删库唯一依据。
- `topic_prefix` 若需匹配特定业务提示模板（如固定主持人话术），需在 `configs/default.yaml` 中补充对应 `explicit_cues` 与 `fillers`。
- 源码遵循 [Apache-2.0](./LICENSE) 开源协议，不包含任何第三方源码拷贝、预训练权重或真实语料。
