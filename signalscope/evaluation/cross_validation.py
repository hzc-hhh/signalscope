"""
Cross-validation strategies aware of temporal structure and subject grouping.

Standard k-fold CV assumes i.i.d. data, which is violated in sensor time-series
and multi-subject biomedical data. SignalScope provides:

- TimeSeriesSplit: Forward-chaining (rolling origin) for temporal data
- SubjectWiseSplit: Leave-subjects-out for multi-subject datasets
"""

from typing import Iterator, List, Optional, Tuple

import numpy as np


class TimeSeriesSplit:
    """
    Time-series aware cross-validation (forward chaining).

    Unlike standard k-fold, never uses future data for training.
    Each split has more training data than the previous one.

    Parameters
    ----------
    n_splits : int
        Number of train/test splits.
    gap : int
        Number of samples to exclude between train and test (prevents leakage).
    test_size : int
        Number of samples in each test fold (default: total / n_splits).
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        test_size: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size

    def split(self, data: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test index splits.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, ...)

        Yields
        ------
        (train_indices, test_indices) : tuple of np.ndarray
        """
        n = len(data)
        test_size = self.test_size or max(1, n // (self.n_splits + 1))

        for i in range(self.n_splits):
            test_end = n - (self.n_splits - i - 1) * test_size
            test_start = test_end - test_size
            train_end = max(0, test_start - self.gap)

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_splits


class SubjectWiseSplit:
    """
    Leave-subject(s)-out cross-validation.

    Essential for biomedical sensor data where multiple samples
    come from the same subject (violating i.i.d. assumption).

    Parameters
    ----------
    n_splits : int
        Number of folds.
    leave_out : int
        Number of subjects to leave out per fold.
    shuffle : bool
        Shuffle subject order.
    random_state : int
        Random seed for shuffling.
    """

    def __init__(
        self,
        n_splits: int = 5,
        leave_out: int = 1,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.leave_out = leave_out
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self,
        subject_ids: List[str],
    ) -> Iterator[Tuple[List[str], List[str]]]:
        """
        Generate subject-wise train/test splits.

        Parameters
        ----------
        subject_ids : list[str]
            Subject identifiers for each sample.

        Yields
        ------
        (train_subjects, test_subjects) : tuple of list[str]
        """
        unique = list(set(subject_ids))
        n_subjects = len(unique)

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(unique)

        fold_size = max(self.leave_out, n_subjects // self.n_splits)

        for i in range(self.n_splits):
            start = i * fold_size
            end = min(start + fold_size, n_subjects)

            test_subjects = unique[start:end]
            train_subjects = [s for s in unique if s not in test_subjects]

            if not test_subjects:
                break
            yield train_subjects, test_subjects

        if self.shuffle:
            self.random_state += 1  # Different shuffle per call

    def get_n_splits(self) -> int:
        return self.n_splits
