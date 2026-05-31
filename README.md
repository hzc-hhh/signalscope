# SignalScope

<div align="center">

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Status](https://img.shields.io/badge/status-alpha-orange)
[![CI](https://github.com/signalscope/signalscope/actions/workflows/ci.yml/badge.svg)](https://github.com/signalscope/signalscope/actions/workflows/ci.yml)

**Unified AI Research Framework for Biomedical Sensor Signals**

*Preprocessing → Model Zoo → Benchmark → Research Pipeline — all in one place.*

</div>

---

## 🎯 The Problem

Biomedical sensor signals — especially **radar-based contactless sensing** — represent a frontier in AI-for-Health. Yet unlike computer vision or NLP, this field lacks:

- **Standardized preprocessing** toolchains for raw sensor signals (IQ streams → physiological waveforms)
- **Unified benchmarks** to compare ML/DL methods fairly
- **Accessible model zoos** bridging classical signal processing to modern SSL/Transformers
- **Reproducible research pipelines** that connect literature survey → idea → experiment → paper

Researchers across labs reinvent the same signal processing wheels, slowing down progress in vital areas like non-contact patient monitoring, sleep analysis, and emergency triage.

## ✅ What SignalScope Provides

| Module | What It Does |
|--------|-------------|
| **Preprocessing** | IQ-to-displacement, clutter removal, phase unwrapping, vital sign extraction for radar; alignment for multi-modal PPG/ECG |
| **Model Zoo** | Classical DSP → 1D CNN/ResNet → Time-Series Transformers → Self-Supervised Pretraining → Multi-Modal Fusion |
| **Evaluation** | Domain-aware metrics (heart rate error, respiration MAE), statistical tests, cross-validation aware of temporal structure |
| **Benchmark** | Standardized dataset interfaces, reproducible runner, leaderboard generation |
| **Interpretability** | Signal attribution analysis, medical semantic mapping from learned representations |
| **Research Pipeline** | Optional AI-assisted workflow: literature survey → idea generation → experiment design → paper draft (human-in-the-loop) |

## 🚀 Quick Start

```bash
pip install signalscope
```

### 5-line example: Radar vital signs extraction

```python
from signalscope.preprocessing.radar import IQPipeline

pipeline = IQPipeline(clutter_filter="adaptive", phase_method="dacm")
iq_data = load_your_radar_data()  # shape: (n_samples, n_chirps)

vital_signs = pipeline(iq_data, sample_rate=1000)
# vital_signs.respiration  → breathing waveform
# vital_signs.heartbeat    → cardiac waveform
```

### Benchmark your model

```python
from signalscope.benchmark import BenchmarkRunner

runner = BenchmarkRunner(
    tasks=["heart_rate", "respiration_rate"],
    models=["resnet1d", "transformer_ts", "ssl_pretrain"],
)
results = runner.run()
results.leaderboard()
```

## 📦 Installation

```bash
# From PyPI
pip install signalscope

# From source (dev)
git clone https://github.com/signalscope/signalscope.git
cd signalscope
pip install -e ".[dev,docs]"
```

## 🧪 Project Structure

```
signalscope/
├── signalscope/           # Main package
│   ├── core/              # Pipeline base, model registry
│   ├── preprocessing/     # Radar, PPG, ECG, multi-modal sync
│   ├── models/            # Classical + Deep (ResNet1D, TransformerTS, SSL)
│   ├── evaluation/        # Metrics, statistics, cross-validation
│   ├── benchmark/         # Dataset interface, runner, leaderboard
│   ├── interpretability/  # Signal attribution, medical mapping
│   └── utils/             # Logging, visualization, config
├── examples/              # Jupyter notebooks & Python scripts
├── tests/                 # Unit & integration tests
└── docs/                  # Full documentation (MkDocs)
```

## 🌍 Why This Matters for Open Source

> The biomedical AI community has excellent tools for images (MONAI, nnU-Net) and text (HuggingFace), but **sensor signal AI remains underserved**. SignalScope is built by researchers who face this gap daily — we are abstracting years of lab-level preprocessing work into reusable, documented, and tested open-source components. Whether you work with radar, PPG, ECG, or inertial sensors, SignalScope gives you a **unified entry point** backed by a standard benchmark so the community can compare methods and reproduce results.

## 📖 Documentation

Full documentation at [signalscope.github.io](https://signalscope.github.io):
- [Getting Started](https://signalscope.github.io/getting-started)
- [Sensor Preprocessing Guide](https://signalscope.github.io/sensor-preprocessing)
- [Model Zoo Reference](https://signalscope.github.io/model-zoo)
- [Benchmark Guide](https://signalscope.github.io/benchmark)
- [API Reference](https://signalscope.github.io/api)

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where you can help:
- Add support for new sensor modalities (IMU, EEG, SpO2, ...)
- Contribute new model architectures
- Share benchmark results on public datasets
- Improve documentation and tutorials

## 📄 Citation

If you use SignalScope in your research, please cite:

```bibtex
@software{signalscope2026,
  title = {SignalScope: Unified AI Research Framework for Biomedical Sensor Signals},
  url = {https://github.com/signalscope/signalscope},
  year = {2026},
}
```

## 📜 License

Apache 2.0 — see [LICENSE](LICENSE) for details.
