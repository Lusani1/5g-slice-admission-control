"""Data loading utilities for Materna-derived synthetic slice profiles."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence
import tempfile
import zipfile
import shutil
import pandas as pd
import numpy as np

FEATURE_COLUMNS = ["cpu_factor", "mem_factor", "bw_factor", "demand_score"]
RESOURCE_COLUMNS = ["cpu_factor", "mem_factor", "bw_factor"]


@dataclass
class ProfileSegment:
    profile_id: str
    frame: pd.DataFrame

    @property
    def features(self) -> np.ndarray:
        return self.frame[FEATURE_COLUMNS].to_numpy(dtype=float)

    @property
    def resource_factors(self) -> np.ndarray:
        return self.frame[RESOURCE_COLUMNS].to_numpy(dtype=float)

    @property
    def demand_score(self) -> np.ndarray:
        return self.frame["demand_score"].to_numpy(dtype=float)

    @property
    def demand_regime(self) -> np.ndarray:
        return self.frame.get("demand_regime", pd.Series(["medium"] * len(self.frame))).astype(str).to_numpy()

    @property
    def is_burst(self) -> np.ndarray:
        if "is_burst" in self.frame.columns:
            return self.frame["is_burst"].astype(bool).to_numpy()
        return np.zeros(len(self.frame), dtype=bool)


class MaternaProfileStore:
    """Loads and samples Materna-derived synthetic slice profiles.

    The loader accepts either the extracted dataset directory or the ZIP package
    produced by the preprocessing pipeline. It expects a `profiles_clean` folder
    containing `clean_profile_*.csv.gz` files.
    """

    def __init__(self, data_root: str | Path, max_profiles: int | None = None, min_rows: int = 1000):
        self.original_data_root = Path(data_root)
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self.data_root = self._prepare_root(self.original_data_root)
        self.profile_paths = self._find_profile_paths(self.data_root)
        if max_profiles is not None and max_profiles > 0:
            self.profile_paths = self.profile_paths[:max_profiles]
        self.min_rows = min_rows
        self._cache: dict[Path, pd.DataFrame] = {}
        if not self.profile_paths:
            raise FileNotFoundError(
                f"No profiles_clean/clean_profile_*.csv.gz files found under {self.original_data_root}"
            )

    def _prepare_root(self, p: Path) -> Path:
        if p.is_file() and p.suffix == ".zip":
            self._tmpdir = tempfile.TemporaryDirectory(prefix="materna_profiles_")
            with zipfile.ZipFile(p, "r") as z:
                z.extractall(self._tmpdir.name)
            return Path(self._tmpdir.name)
        return p

    @staticmethod
    def _find_profile_paths(root: Path) -> List[Path]:
        paths = sorted(root.rglob("profiles_clean/clean_profile_*.csv.gz"))
        if not paths:
            paths = sorted(root.rglob("clean_profile_*.csv.gz"))
        return paths

    def close(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def __len__(self) -> int:
        return len(self.profile_paths)

    def load_profile(self, path: Path) -> pd.DataFrame:
        if path in self._cache:
            return self._cache[path]
        df = pd.read_csv(path)
        required = set(FEATURE_COLUMNS)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Profile {path} is missing columns {sorted(missing)}")
        # Normalise booleans and keep only rows with finite feature values.
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        self._cache[path] = df
        return df

    def iter_profiles(self, max_profiles: int | None = None) -> Iterable[pd.DataFrame]:
        paths = self.profile_paths[:max_profiles] if max_profiles else self.profile_paths
        for p in paths:
            df = self.load_profile(p)
            if len(df) >= self.min_rows:
                yield df

    def sample_segment(
        self,
        rng: np.random.Generator,
        length: int,
        min_length: int | None = None,
        split_column: str | None = None,
        split_value: str | None = None,
        allow_fallback: bool = False,
    ) -> tuple[ProfileSegment, bool]:
        """Sample a contiguous workload segment.

        If ``split_column`` and ``split_value`` are supplied, candidate rows
        are restricted to that chronological split. If no profile has enough
        contiguous rows, optional fallback to the full profile is reported via
        ``used_full_profile_fallback``. The reference experiment disables this
        fallback.
        """
        min_len = min_length or length
        used_fallback = False

        def _candidates(filter_split: bool) -> list[tuple[Path, pd.DataFrame]]:
            out = []
            for p in self.profile_paths:
                df = self.load_profile(p)
                if filter_split and split_column and split_value and split_column in df.columns:
                    df = df[df[split_column].astype(str).str.lower() == str(split_value).lower()].reset_index(drop=True)
                if len(df) >= min_len:
                    out.append((p, df))
            return out

        candidates = _candidates(filter_split=bool(split_column and split_value))
        if not candidates and split_column and split_value:
            if not allow_fallback:
                raise ValueError(
                    f"No profile has at least {min_len} contiguous rows for "
                    f"{split_column}={split_value!r}. Refusing full-profile fallback "
                    "because strict split sampling is enabled."
                )
            # Explicit ablation/debug mode only: fall back to the full profile and flag it.
            used_fallback = True
            candidates = _candidates(filter_split=False)
        if not candidates:
            raise ValueError(f"No profile with at least {min_len} rows available")

        p, df = candidates[int(rng.integers(0, len(candidates)))]
        if len(df) <= length:
            start = 0
            seg = df.copy()
        else:
            start = int(rng.integers(0, len(df) - length))
            seg = df.iloc[start:start + length].reset_index(drop=True).copy()
        profile_id = str(seg["slice_profile_id"].iloc[0]) if "slice_profile_id" in seg.columns and len(seg) else p.stem
        return ProfileSegment(profile_id=profile_id, frame=seg), used_fallback

    def build_supervised_windows(
        self,
        window: int,
        horizon: int,
        max_windows: int = 250_000,
        split: str = "split_80_20",
        split_value: str = "train",
        rng: np.random.Generator | None = None,
        max_profiles: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build flattened lag-window samples for XGBoost.

        The default uses rows labelled as `train` in `split_80_20`, matching the
        final-reporting train split in the journal methodology.
        """
        rng = rng or np.random.default_rng(42)
        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        n_total = 0
        for df in self.iter_profiles(max_profiles=max_profiles):
            if split in df.columns:
                dfx = df[df[split].astype(str).str.lower() == split_value.lower()].reset_index(drop=True)
                if len(dfx) < window + horizon + 1:
                    continue
            else:
                cut = int(len(df) * 0.8)
                dfx = df.iloc[:cut].reset_index(drop=True)
            features = dfx[FEATURE_COLUMNS].to_numpy(dtype=float)
            targets = dfx[RESOURCE_COLUMNS].to_numpy(dtype=float)
            n = len(dfx) - window - horizon + 1
            if n <= 0:
                continue
            # Avoid huge memory spikes by subsampling per profile if needed.
            indices = np.arange(n)
            remaining = max_windows - n_total if max_windows and max_windows > 0 else n
            if remaining <= 0:
                break
            if max_windows and n > remaining:
                indices = np.sort(rng.choice(indices, size=remaining, replace=False))
            Xp = np.empty((len(indices), window * len(FEATURE_COLUMNS)), dtype=np.float32)
            yp = np.empty((len(indices), len(RESOURCE_COLUMNS)), dtype=np.float32)
            for row, idx in enumerate(indices):
                Xp[row] = features[idx:idx + window].reshape(-1)
                yp[row] = targets[idx + window + horizon - 1]
            X_parts.append(Xp)
            y_parts.append(yp)
            n_total += len(indices)
            if max_windows and n_total >= max_windows:
                break
        if not X_parts:
            raise ValueError("No supervised windows could be built from the profiles")
        X = np.vstack(X_parts)
        y = np.vstack(y_parts)
        return X, y


def ensure_dataset_extracted(data_root: str | Path, out_dir: str | Path) -> Path:
    """Extract a dataset ZIP to a stable directory if needed."""
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    if data_root.is_dir():
        return data_root
    if not data_root.exists():
        raise FileNotFoundError(data_root)
    if out_dir.exists() and list(out_dir.rglob("profiles_clean/clean_profile_*.csv.gz")):
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(data_root, "r") as z:
        z.extractall(out_dir)
    return out_dir
