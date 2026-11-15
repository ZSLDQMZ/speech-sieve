[English](./README.md) | [简体中文](./README_zh.md)

# speech-sieve

**speech-sieve** is a modular, **Apache-2.0** licensed speech dataset quality profiling and alignment auditing toolkit. Given an audio manifest (`JSONL`), it outputs per-utterance acoustic metrics, optional voice-activity and forced-alignment checks, overlap and prompt-echo signals, and routes records into **`normal`**, **`review_recommended`**, or **`alert`**.

All core signal operators depend only on the Python standard library, `NumPy`, and `PyYAML`, so you can run waveform checks and acoustic repetition detection out of the box on CPU. Optional backends (Silero VAD, Qwen3-ForcedAligner, and `pyannote` overlapped-speech detection) load lazily when enabled.

---

## Why `speech-sieve` (Problems Solved & Use Cases)

When curating real-world speech datasets for **TTS, voice cloning, ASR, or speech LLMs** (from crowdsourced reading, audiobook segmentation, or interviews), off-the-shelf sanity scripts often miss subtle issues or over-filter good recordings:

1. **Subtle recording defects escape simple duration/volume checks**: Speakers stumble and re-read a phrase within the same take (stutters or mid-take restarts), background voices leak in (overlapped speech), unscripted chatter trails before or after the prompt, or interviewees accidentally read the on-screen question prompt aloud before answering.
2. **Rigid global thresholds and ASR-CER cutoffs discard high-value speech**: For accented, dialectal, expressive, or polyphonic utterances, general-purpose ASR models often misrecognize valid pronunciations and produce high Character Error Rates (CER). Dropping recordings on a fixed CER cutoff removes the exact rare data you want to preserve; likewise, turning speech rhythm or spectral brightness into binary pass/fail rules causes false positives on short clips.
3. **Missing checks silently masquerade as passes**: Pipelines that substitute `0.0` when an optional model is disabled or when an utterance is too short for window statistics hide gaps in quality coverage.

**Common use cases**:
- Pre-ingestion quality screening and human-review prioritization for scripted TTS / reading corpora;
- Speaker overlap, clipping, and prompt read-aloud screening for interviews and unscripted speech;
- Standalone Python/NumPy signal profiling and acoustic self-repetition building blocks embedded inside larger data pipelines.

---

## Pipeline Overview & Implementation Design

```text
Input Manifest (manifest.jsonl: audio_path, reference_text?, question?)
   │
   ├─► [1] Audio Loading & Waveform Integrity (audio.py, features.py)
   │        └─ Decoding fallbacks, channel/rate policy, empty & clipping checks
   │
   ├─► [2] Multi-Dimensional Acoustic Profiling (features.py)
   │        └─ Rate-adaptive framing, level dynamics, pause cadence, 2048-pt FFT spectrum
   │
   ├─► [3] Unsupervised Acoustic Self-Repetition Detection (repeat.py)
   │        └─ Multi-scale downsampled MFCC coarse search + subsequence DTW verification
   │
   ├─► [4] VAD & Forced-Alignment Verification [Optional] (models.py, alignment.py)
   │        └─ Silero VAD + Qwen3-ForcedAligner span coverage & text-repeat cross-checking
   │
   ├─► [5] Overlapped Speech & Prompt Read-Aloud Detection [Optional] (models.py, topic_prefix.py)
   │        └─ Batched pyannote sliding-window inference + opening prompt overlap matching
   │
   └─► [6] Explicit Status Reporting & Triage Routing (pipeline.py, cli.py)
            └─ Length-bucketed batching & OOM halving -> report.jsonl (normal / review_recommended / alert)
```

### Stage-by-Stage Implementation

#### 1. Cross-Platform Audio Loading & Integrity (`audio.py`, `features.py`)
- **Multi-backend decoding**: Tries `soundfile`, falls back to Python stdlib `wave`, and then `scipy.io.wavfile`, supporting standard PCM WAV files without requiring extra native libraries.
- **Flexible format enforcement**: Can automatically downmix to mono and resample to a target sample rate when aggregating heterogeneous sources, or enforce strict mono/sample-rate matching for controlled recording campaigns.
- **Hard signal guards**: Catches zero-duration/all-zero recordings and measures near-full-scale sample clipping before downstream processing.

#### 2. Sample-Rate-Adaptive Acoustic Profiling (`features.py`)
- **Adaptive framing & Nyquist-safe spectral bands**: Short-time frame and hop sizes are derived dynamically from each file's sample rate, and spectral energy bands are clamped against the Nyquist limit so 8 kHz narrowband recordings are analyzed cleanly alongside 16 kHz+ audio.
- **Separating hard failures from continuous descriptors**:
  - Computes active speech digital signal level (`speech_rms_dbfs`), multi-window dynamic spread (`P90 - P10` dB), dip-depth pause counts, and high-resolution **2048-point FFT** spectral rolloff and sub-band ratios;
  - By default, rhythm cadence and spectral tilt are emitted as continuous profiling telemetry rather than hard rejection gates, and rate-based metrics require a minimum voiced duration so short clips are never distorted by division over small durations.

#### 3. ASR-Free Acoustic Self-Repetition Detection (`repeat.py`)
- **Problem addressed**: When a speaker flubs a word and immediately repeats the phrase, the reference transcript usually only contains the clean sentence once—making the extra take invisible to text-only checks.
- **Method**: Extracts Mel-frequency cepstral coefficients (MFCC) directly from the audio, builds downsampled block descriptors at multiple seed lengths (`0.48 s`, `0.96 s`, `1.92 s`) for fast similarity retrieval, and validates top candidates with **Subsequence Dynamic Time Warping (DTW)**. Time-warping supports speed variations between takes (`0.67×–1.5×`) while bounding compute per file with an explicit runtime cap.

#### 4. VAD & Forced-Alignment Cross-Checking (`models.py`, `alignment.py`, Optional)
- **Span coverage checks**: Pairs Silero VAD speech intervals with Qwen3-ForcedAligner word/character spans to detect unexplained extra speech outside the aligned transcript, excessive silence, or very low token-per-speech ratios.
- **Acoustic-text repetition reconciliation**: Cross-references acoustic self-repetition candidates from Stage 3 against forced-alignment intervals. If the reference transcript itself contains intentional repeated words or parallel phrasing at those timestamps, the match is cleared as expected repetition; unaligned voiced repetitions are flagged for triage.
- **Language-agnostic token estimation**: Automatically switches between whitespace tokenization, character counting, and mixed-script counting via `token_unit: auto | char | whitespace`.

#### 5. Overlapped Speech & Prompt Read-Aloud Plugins (`models.py`, `topic_prefix.py`, Optional)
- **Batched sliding-window overlap inference (`BatchedOverlapAdapter`)**: Wraps `pyannote/segmentation-3.0` and packs sliding windows across multiple utterances into unified GPU batches for high throughput, while preserving valid interpolation weights on short utterances (`< 10 s`) padded on the right.
- **Prompt read-aloud detection (`topic_prefix.py`)**: Compares the opening clause of an utterance's text against an optional `question` prompt field using configurable prefix stripping, filler skipping, and `SequenceMatcher` fuzzy overlap so recordings where speakers read the question aloud are easily isolated.

#### 6. Explicit Per-Gate Status & Resilient Batching (`pipeline.py`, `cli.py`)
- **Three explicit states per check**: Every gate reports `evaluated` (ran with met preconditions), `not_evaluated` (optional model disabled or utterance too short/narrowband, with reason code), or `skipped` (disabled for the current operating mode, such as skipping transcript alignment checks in `spontaneous` mode).
- **Robust scheduling**: Sorts batch items into length buckets to reduce tensor padding waste and automatically halves batch size on CUDA out-of-memory (OOM) errors so one difficult utterance never aborts an entire batch.

---

## Installation & Quickstart

### 1. Install

Core installation (CPU-only, requires Python >= 3.10, `NumPy`, and `PyYAML`):

```bash
pip install -e .
```

Install optional backends as needed:

```bash
pip install -e ".[soundfile]"   # Extra audio codecs via libsndfile (stdlib wave works without it)
pip install -e ".[vad]"         # Silero VAD (MIT)
pip install -e ".[align]"       # Qwen3-ForcedAligner-0.6B via qwen-asr (Apache-2.0)
pip install -e ".[overlap]"     # pyannote.audio overlap detection (MIT; gated Hugging Face repo)
```

> **Note on `[overlap]`**: `pyannote/segmentation-3.0` is a gated Hugging Face repository. Accept its terms on the Hugging Face model page and configure your `HF_TOKEN` (or point to a local cached checkpoint) before enabling it. See [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).

### 2. Run the Synthetic Demo (Zero Real Audio or GPU Needed)

On Windows PowerShell, set `$env:PYTHONPATH="src"` before running:

```bash
# 1. Generate 5 synthetic WAV fixtures (clean, clipped, prompt-echo, repeated, 8kHz stereo) + manifest.jsonl
PYTHONPATH=src python examples/generate_demo.py --out-dir demo_data

# 2. Validate the YAML configuration
PYTHONPATH=src python -m speech_sieve.cli check --config configs/default.yaml

# 3. Run the CPU audit pipeline and write report.jsonl
PYTHONPATH=src python -m speech_sieve.cli run \
  --input demo_data/manifest.jsonl \
  --config configs/default.yaml \
  --output demo_data/report.jsonl
```

Run the standard library `unittest` suite:

```bash
PYTHONPATH=src python -B -m unittest discover -s tests -v
```

---

## Configuration & Adaptation

`speech-sieve` uses a single neutral default configuration file, [`configs/default.yaml`](./configs/default.yaml), with inline comments explaining every section. Copy it to adapt to your corpus:

| Target Corpus | Suggested Adjustments in `default.yaml` | Behavior |
| --- | --- | --- |
| **Quick CPU screening** | Use `configs/default.yaml` as-is | Zero heavy dependencies; runs format normalization, clipping/level checks, acoustic repetition DTW, and prompt overlap checks |
| **Scripted reading / TTS curation** | Keep `mode: scripted_reading`; enable `models.vad`, `models.aligner`, and `alignment` (optionally `overlap`); set `audio.require_mono: true` if strict recording specs are required | Enables VAD-to-alignment coverage checking, silence ratios, and acoustic-vs-text repeat reconciliation |
| **Spontaneous speech / interviews** | Set `mode: spontaneous`; enable `vad` and `overlap`; customize `topic_prefix.explicit_cues` and `fillers` if speakers use recurring prompt templates | Automatically skips transcript-alignment gates while catching overlapped speech and prompt read-alouds |

---

## Repository Layout & Scope Notes

```text
.
├── pyproject.toml                  # Package metadata (speech-sieve / speech_sieve) & optional extras
├── LICENSE                         # Apache License 2.0
├── THIRD_PARTY_NOTICES.md          # Dependency licenses & gated model instructions
├── README.md                       # English documentation
├── README_zh.md                    # Simplified Chinese documentation
├── configs/
│   └── default.yaml                # Single neutral default configuration with inline comments
├── src/speech_sieve/
│   ├── __init__.py
│   ├── audio.py                    # Multi-backend audio decoding, mono mixdown & resampling
│   ├── config.py                   # YAML loading, physical invariant validation & atomic I/O
│   ├── features.py                 # Integrity, speech RMS dynamics, dip cadence & FFT spectrum
│   ├── repeat.py                   # Multi-scale MFCC + subsequence DTW repetition detector
│   ├── alignment.py                # Language-adaptive token counting, coverage & repeat cross-check
│   ├── topic_prefix.py             # Configurable prompt read-aloud / opening-cue detector
│   ├── models.py                   # Lazy adapters (VAD / Aligner / BatchedOverlap) & OOM halving
│   ├── pipeline.py                 # Tri-state gate evaluation & triage routing
│   └── cli.py                      # check / run CLI entrypoint
├── examples/
│   └── generate_demo.py            # Synthetic WAV fixture & manifest generator
└── tests/
    ├── test_operators.py
    ├── test_gating_and_config.py
    └── test_mock_pipeline.py
```

**Scope Notes**:
- Digital RMS (`dBFS`) measures waveform signal amplitude and does not replace acoustic SPL or perceptual speech quality metrics (MOS/PESQ).
- Signals such as `overlap_risk`, `prompt_echo_prefix`, and optional `advisory_cer` are intended to prioritize human review rather than serve as sole automatic deletion cutoffs.
- Licensed under [Apache-2.0](./LICENSE); contains zero third-party source code, pretrained weights, or private audio recordings.
