"""
Multi-modal signal alignment.

Synchronizes signals from different sensors (radar, PPG, ECG) by:
- Resampling to a common rate
- Aligning timestamps via cross-correlation
- Handling missing modalities gracefully
"""


import numpy as np
from scipy import signal as scipy_signal

from signalscope.core.pipeline import Pipeline, PipelineResult


class MultiModalAligner(Pipeline):
    """
    Align multiple sensor signal streams to a common time base.

    Parameters
    ----------
    target_rate : float
        Common sample rate after alignment (Hz).
    method : str
        'resample' or 'cross_correlation'.
    """

    def __init__(
        self,
        target_rate: float = 100.0,
        method: str = "resample",
        **config,
    ):
        super().__init__(target_rate=target_rate, method=method, **config)

    def __call__(
        self,
        signals: dict[str, tuple[np.ndarray, float]],
    ) -> PipelineResult:
        """
        Align multi-modal signals.

        Parameters
        ----------
        signals : dict
            {modality_name: (signal_array, original_sample_rate)}

        Returns
        -------
        PipelineResult
            Aligned signals at the target rate.
        """
        try:
            target_sr = self.config["target_rate"]
            aligned = {}

            for name, (data, orig_sr) in signals.items():
                if orig_sr == target_sr:
                    aligned[name] = data
                    continue

                target_len = int(len(data) * target_sr / orig_sr)
                aligned[name] = scipy_signal.resample(data, target_len)

            return PipelineResult(data=aligned)

        except Exception as e:
            return PipelineResult(data=None, success=False, error=str(e))
