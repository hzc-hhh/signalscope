# Contributing to SignalScope

Thank you for your interest in contributing! SignalScope is an open-source project aiming to build a unified AI research framework for biomedical sensor signals.

## 🌟 Ways to Contribute

### 🆕 Add a New Sensor Modality
We want SignalScope to support as many biomedical sensor types as possible (EEG, EMG, IMU, SpO2, etc.). See our [sensor contribution template](.github/ISSUE_TEMPLATE/new_sensor_template.md).

### 🧠 Contribute a Model
Add your model to the Model Zoo:
1. Implement the model class inheriting from `signalscope.models.base.BaseModel`
2. Register it in `signalscope/models/zoo.py`
3. Add unit tests
4. Submit a PR with a benchmark comparison on at least one public dataset

### 🐛 Report Bugs
Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md). Include:
- SignalScope version
- Python version and OS
- Minimal reproducible example
- Expected vs actual behavior

### 📝 Improve Documentation
Documentation fixes, tutorials, and examples are always welcome.

## 🔧 Development Setup

```bash
# Clone and install in dev mode
git clone https://github.com/signalscope/signalscope.git
cd signalscope
pip install -e ".[dev,docs]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Lint
ruff check signalscope/
```

## 📋 Pull Request Process

1. **Fork** the repository
2. Create a **feature branch** (`git checkout -b feature/your-feature`)
3. **Write tests** for new functionality
4. Ensure **CI passes** (`pytest && ruff check`)
5. Update **documentation** if needed
6. Submit a **PR** with a clear description

## 🏷️ Commit Convention

We use conventional commits:
- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation
- `test:` — tests
- `refactor:` — code restructuring
- `chore:` — maintenance

## 📜 Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## ❓ Questions?

Open a [discussion](https://github.com/signalscope/signalscope/discussions) or ask in an issue.
