"""
Tests for SignalScope preprocessing module.
"""

import numpy as np
import pytest

from signalscope.preprocessing.radar import (
    IQPipeline,
    VitalSigns,
    remove_clutter,
    estimate_bpm,
)


class TestIQPipeline:
    def test_pipeline_creation(self):
        pipeline = IQPipeline(clutter_filter="adaptive", phase_method="dacm")
        assert pipeline.config["clutter_filter"] == "adaptive"
        assert pipeline.config["phase_method"] == "dacm"

    def test_pipeline_runs_on_synthetic_data(self):
        pipeline = IQPipeline(sample_rate=1000)

        t = np.linspace(0, 10, 10000)
        iq = np.sin(2 * np.pi * 1.2 * t) + 0.05 * np.random.randn(10000)

        result = pipeline(iq, sample_rate=1000)
        assert result.success
        assert isinstance(result.data, VitalSigns)
        assert result.data.respiration.shape == (10000,)
        assert result.data.heartbeat.shape == (10000,)

    def test_pipeline_handles_multichannel(self):
        pipeline = IQPipeline(sample_rate=500)
        data = np.random.randn(5000, 4)  # 4 chirps per frame

        result = pipeline(data, sample_rate=500)
        assert result.success
        assert result.data.respiration is not None


class TestSignalUtils:
    def test_static_clutter(self):
        data = np.ones((100,)) + 0.1 * np.random.randn(100)
        cleaned = remove_clutter(data, method="static")
        assert cleaned.shape == data.shape
        assert abs(np.mean(cleaned)) < 0.1

    def test_adaptive_clutter(self):
        data = np.sin(np.linspace(0, 4 * np.pi, 200)) + 5.0  # DC offset
        cleaned = remove_clutter(data, method="adaptive", alpha=0.95)
        assert cleaned.shape == data.shape

    def test_estimate_bpm(self):
        sr = 1000
        t = np.linspace(0, 10, 10 * sr)
        signal = np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz = 72 BPM
        bpm = estimate_bpm(signal, sr, freq_range=(0.8, 3.0))
        assert bpm is not None
        assert 60 < bpm < 85


class TestPPGProcessor:
    def test_ppg_creation(self):
        from signalscope.preprocessing.ppg import PPGProcessor
        proc = PPGProcessor(sample_rate=100)
        assert proc.config["sample_rate"] == 100

    def test_ppg_runs(self):
        from signalscope.preprocessing.ppg import PPGProcessor
        proc = PPGProcessor(sample_rate=100)
        sr = 100
        t = np.linspace(0, 10, 10 * sr)
        ppg = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 0.3 * t)
        result = proc(ppg, sample_rate=sr)
        assert result.success
