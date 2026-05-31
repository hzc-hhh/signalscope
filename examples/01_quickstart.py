"""SignalScope quickstart example."""

import numpy as np
from signalscope.preprocessing.radar import IQPipeline
from signalscope.models import ModelZoo
from signalscope.benchmark import BenchmarkRunner
from signalscope.utils import plot_signal


def main():
    # --- 1. Preprocessing: Simulate radar IQ signal ---
    print("=" * 60)
    print("SignalScope Quickstart")
    print("=" * 60)

    print("\n[1] Radar IQ preprocessing...")
    pipeline = IQPipeline(
        clutter_filter="adaptive",
        phase_method="dacm",
        bandpass=(0.1, 3.0),
        sample_rate=1000,
    )

    # Simulate IQ data: a sine wave with noise
    t = np.linspace(0, 10, 10000)
    iq_data = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(10000)

    result = pipeline(iq_data, sample_rate=1000)
    if result.success:
        vs = result.data
        print(f"   Respiration shape: {vs.respiration.shape}")
        print(f"   Heartbeat shape:   {vs.heartbeat.shape}")
        print(f"   Estimated HR:      {vs.heart_rate_bpm:.1f} BPM")
        print(f"   Estimated RR:      {vs.respiration_rate_bpm:.1f} brpm")
    else:
        print(f"   Error: {result.error}")

    # --- 2. Model Zoo ---
    print("\n[2] Model Zoo...")
    zoo = ModelZoo()
    print(zoo.summary())

    model = zoo.get("resnet1d", in_channels=1, num_classes=1)
    print(f"   Created: {model}")

    # --- 3. Benchmark ---
    print("\n[3] Running benchmark...")
    runner = BenchmarkRunner(
        tasks=["heart_rate"],
        models=["classical", "resnet1d", "transformer_ts"],
        metrics=["mae"],
        n_splits=3,
    )
    bench_result = runner.run()
    if bench_result.success:
        print(bench_result.data)

    print("\n✓ Done! Explore more at https://github.com/signalscope/signalscope")


if __name__ == "__main__":
    main()
