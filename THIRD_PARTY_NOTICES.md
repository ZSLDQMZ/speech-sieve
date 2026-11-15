# Third-Party Notices

## Original Source & Zero Bundled Assets Declaration

This repository (**`speech-sieve`** / package `speech_sieve`) contains **100% original source code** authored for modular speech quality auditing and alignment-based corpus verification.

- **Zero copied third-party source code**: No source files from external models, aligners, or signal-processing libraries are copied or vendored into this repository.
- **Zero bundled model weights**: No pretrained checkpoints, model weights, or authentication tokens are included in this repository.
- **Zero copyrighted or private corpora**: No real-world speech recordings, speaker identifiers, or proprietary datasets are included. All examples and tests run on programmatically synthesized audio waveforms.

All original code in this repository is licensed under the **Apache License, Version 2.0** (see [`LICENSE`](./LICENSE)).

---

## Core & Optional Third-Party Components

Core signal operators, configuration loading, and pipeline gating depend only on the Python standard library, `NumPy`, and `PyYAML`. Heavy neural backends and extra audio codecs are strictly **optional** (`extras_require`) and are loaded lazily at runtime only when explicitly enabled.

| Component | Role in `speech-sieve` | License | Distribution / Usage Notes |
| --- | --- | --- | --- |
| **NumPy** | Core array math, FFT, MFCC/DTW, signal descriptors | **BSD-3-Clause** | Core runtime dependency (installed separately by user). |
| **PyYAML** | YAML preset & schema configuration parsing | **MIT** | Core runtime dependency (installed separately by user). |
| **SciPy** | Optional WAV decoding fallback (`scipy.io.wavfile`) & signal utilities | **BSD-3-Clause** | Optional (`[soundfile]` / fallback). Stdlib `wave` works with zero SciPy/soundfile. |
| **soundfile** | Optional multi-format audio reading (`libsndfile` wrapper) | **BSD-3-Clause** | Optional (`[soundfile]`). Automatically falls back to `wave` / `scipy.io.wavfile` if absent. |
| **Silero VAD** | Optional Voice Activity Detection (`[vad]`) | **MIT** | Loaded at runtime via user-installed `silero-vad`, `torch.hub`, or user-supplied local JIT path. |
| **Qwen3-ForcedAligner-0.6B** (`qwen-asr`) | Optional forced text-audio alignment (`[align]`, via `qwen-asr` + `transformers` + `torch`) | **Apache-2.0** | Loaded at runtime via user-configured Hugging Face model ID or local checkpoint directory. |
| **pyannote.audio** & **`pyannote/segmentation-3.0`** | Optional overlapped-speech risk detection (`[overlap]`) | **MIT** | **Hugging Face Gated Repo** — see mandatory user authorization notice below. |

---

## Important Notice for `pyannote.audio` & `pyannote/segmentation-3.0` (Hugging Face Gated Repo)

`pyannote/segmentation-3.0` is hosted as a **gated repository** on the Hugging Face Hub. Before enabling the `[overlap]` backend with pretrained Hugging Face weights:

1. Create or log in to your own **Hugging Face** account.
2. Visit the official model pages (`pyannote/segmentation-3.0` and related `pyannote` prerequisites) and **explicitly review and accept their usage terms / access conditions**.
3. Authenticate locally via `HF_TOKEN` (environment variable) or `huggingface-cli login`, **or** point the YAML configuration to a locally cached checkpoint directory that you obtained under those terms.

`speech-sieve` does not ship, mirror, cache, or proxy any `pyannote` model weights or Hugging Face access tokens. Users are solely responsible for complying with upstream model licenses and gated-access terms.
