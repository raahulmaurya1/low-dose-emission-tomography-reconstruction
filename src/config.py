# config.py: the Settings dataclass.
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Settings dataclass for tomographic reconstruction experiments.

    Parameters:
        n: Image dimension (n x n), default 128.
        n_angles: Number of projection angles, default 60.
        counts: Total expected photon count dose, default 2e5.
        seed: Random generator seed, default 0.
        n_iter: Number of iterations for iterative reconstruction, default 100.
        n_subsets: Number of subsets for OSEM, default 1.
        beta: Regularization weight for TV penalty, default 0.0.
        lesion: Whether to include a synthetic lesion, default False.
    """
    n: int = 128
    n_angles: int = 60
    counts: float = 2e5
    seed: int = 0
    n_iter: int = 100
    n_subsets: int = 1
    beta: float = 0.0
    lesion: bool = False
