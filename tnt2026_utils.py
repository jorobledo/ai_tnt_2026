from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


CSV_COLUMNS = ["Q", "deltaE", "Total"]
DOS_COLUMNS = ["E", "Total_DOS"]
SPACE_GROUP_NUMBER_RE = re.compile(r"\((\d+)\)")
KB_MEV_PER_K = 0.08617333262


def seed_everything(seed: int = 42) -> None:
    """Seed NumPy and PyTorch for reproducible tutorial runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_project_root(start: Path | None = None) -> Path:
    """Walk upward from *start* until the tutorial dataset root is found."""
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "crystals").exists() and (candidate / "crystals.dat").exists():
            return candidate
    raise FileNotFoundError("Could not find the project root with crystals/ and crystals.dat")


def extract_space_group_number(space_group: str) -> int:
    """Extract the numeric space-group identifier from a metadata string."""
    match = SPACE_GROUP_NUMBER_RE.search(space_group)
    if match is None:
        raise ValueError(f"Could not parse space-group number from: {space_group}")
    return int(match.group(1))


def crystal_system_from_space_group(number: int) -> str:
    """Map an international space-group number onto a crystal system."""
    if 1 <= number <= 2:
        return "triclinic"
    if 3 <= number <= 15:
        return "monoclinic"
    if 16 <= number <= 74:
        return "orthorhombic"
    if 75 <= number <= 142:
        return "tetragonal"
    if 143 <= number <= 167:
        return "trigonal"
    if 168 <= number <= 194:
        return "hexagonal"
    if 195 <= number <= 230:
        return "cubic"
    raise ValueError(f"Invalid space-group number: {number}")


def parse_crystals_metadata(metadata_path: str | Path) -> pd.DataFrame:
    """Read `crystals.dat` into a small metadata table keyed by crystal id."""
    rows: list[dict[str, object]] = []
    with Path(metadata_path).open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("-") or line.startswith("id"):
                continue
            parts = re.split(r"\s{2,}", line)
            if len(parts) < 3:
                continue
            crystal_id, formula, space_group = parts[:3]
            sg_number = extract_space_group_number(space_group)
            rows.append(
                {
                    "crystal_id": crystal_id,
                    "formula": formula,
                    "space_group": space_group,
                    "space_group_number": sg_number,
                    "crystal_system": crystal_system_from_space_group(sg_number),
                }
            )
    return pd.DataFrame(rows)


def build_crystal_index(
    crystals_root: str | Path,
    metadata_path: str | Path,
    max_samples: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Join metadata with available spectrum files.

    Each returned row corresponds to one crystal directory containing a
    `powder_2Dmesh_coh_0K.csv` file.
    """
    metadata = parse_crystals_metadata(metadata_path)
    records: list[dict[str, object]] = []
    for csv_path in sorted(Path(crystals_root).glob("*/powder_2Dmesh_coh_0K.csv")):
        crystal_id = csv_path.parent.name
        row = metadata.loc[metadata["crystal_id"] == crystal_id]
        if row.empty:
            continue
        record = row.iloc[0].to_dict()
        record["csv_path"] = csv_path
        records.append(record)

    index = pd.DataFrame(records).sort_values("crystal_id").reset_index(drop=True)
    if max_samples is not None and max_samples < len(index):
        index = index.sample(n=max_samples, random_state=seed).sort_values("crystal_id").reset_index(drop=True)
    return index


def load_spectrum_table(csv_path: str | Path) -> pd.DataFrame:
    """Load a long-form `S(Q,E)` CSV table with columns Q, deltaE, and intensity."""
    table = pd.read_csv(
        csv_path,
        comment="#",
        header=None,
        names=CSV_COLUMNS,
        skipinitialspace=True,
    )
    return table.dropna().reset_index(drop=True)


def load_spectrum_grid(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape a long-form spectrum table into a regular 2D intensity grid."""
    table = load_spectrum_table(csv_path)
    q_values = table["Q"].drop_duplicates().to_numpy(dtype=np.float32)
    e_values = table["deltaE"].drop_duplicates().to_numpy(dtype=np.float32)
    image = table["Total"].to_numpy(dtype=np.float32).reshape(len(q_values), len(e_values))
    return q_values, e_values, image


def resize_image(image: np.ndarray, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Resize a spectrum image with bilinear interpolation."""
    tensor = torch.as_tensor(image, dtype=torch.float32)[None, None, :, :]
    resized = F.interpolate(tensor, size=tuple(size), mode="bilinear", align_corners=False)
    return resized[0, 0].cpu().numpy()


def prepare_image(
    csv_path: str | Path,
    size: tuple[int, int] = (64, 64),
    base_transform: str = "log1p",
    mean: float | None = None,
    std: float | None = None,
) -> np.ndarray:
    """Load one spectrum and convert it into a normalized image array."""
    _, _, image = load_spectrum_grid(csv_path)
    image = resize_image(image, size=size)
    if base_transform == "log1p":
        image = np.log1p(np.clip(image, a_min=0.0, a_max=None))
    elif base_transform != "none":
        raise ValueError(f"Unsupported transform: {base_transform}")
    if mean is not None and std is not None:
        image = (image - mean) / max(std, 1e-6)
    return image.astype(np.float32)


def compute_image_stats(
    csv_paths,
    size: tuple[int, int] = (64, 64),
    base_transform: str = "log1p",
) -> tuple[float, float]:
    """Compute a global mean and standard deviation over a set of images."""
    count = 0
    total = 0.0
    total_sq = 0.0
    for csv_path in csv_paths:
        image = prepare_image(csv_path, size=size, base_transform=base_transform)
        total += float(image.sum())
        total_sq += float(np.square(image).sum())
        count += int(image.size)
    mean = total / count
    variance = max(total_sq / count - mean * mean, 1e-12)
    return mean, math.sqrt(variance)


def make_image_tensor(
    index: pd.DataFrame,
    size: tuple[int, int] = (64, 64),
    base_transform: str = "log1p",
    mean: float | None = None,
    std: float | None = None,
) -> torch.Tensor:
    """Stack many preprocessed spectra into a `(N, 1, H, W)` tensor."""
    images = [
        prepare_image(row.csv_path, size=size, base_transform=base_transform, mean=mean, std=std)
        for row in index.itertuples()
    ]
    array = np.stack(images).astype(np.float32)
    return torch.from_numpy(array[:, None, :, :])


def dataframe_train_test_split(
    index: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    stratify_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a crystal-level table into train and test subsets.

    When `stratify_col` is given, each label group is split separately before
    the pieces are concatenated again.
    """
    rng = np.random.default_rng(seed)
    if stratify_col is None:
        order = rng.permutation(len(index))
        n_test = max(1, int(round(len(index) * test_size)))
        test_idx = np.sort(order[:n_test])
        train_idx = np.sort(order[n_test:])
    else:
        test_parts: list[np.ndarray] = []
        train_parts: list[np.ndarray] = []
        for _, group in index.groupby(stratify_col):
            positions = group.index.to_numpy()
            positions = positions[rng.permutation(len(positions))]
            n_test_group = max(1, int(round(len(positions) * test_size)))
            if len(positions) == 1:
                train_parts.append(positions)
                continue
            if n_test_group >= len(positions):
                n_test_group = len(positions) - 1
            test_parts.append(np.sort(positions[:n_test_group]))
            train_parts.append(np.sort(positions[n_test_group:]))
        test_idx = np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=int)
        train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
    train_df = index.loc[train_idx].reset_index(drop=True)
    test_df = index.loc[test_idx].reset_index(drop=True)
    return train_df, test_df


def load_dos_table(csv_path: str | Path) -> pd.DataFrame:
    """Read the total DOS from `vis_dos.csv`.

    Some files contain extra partial-DOS columns; this reader keeps only the
    first two columns, energy and total DOS.
    """
    rows: list[tuple[float, float]] = []
    with Path(csv_path).open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                energy = float(parts[0])
                total_dos = float(parts[1])
            except ValueError:
                continue
            rows.append((energy, total_dos))
    return pd.DataFrame(rows, columns=DOS_COLUMNS)


def resample_dos_curve(
    energies: np.ndarray,
    dos: np.ndarray,
    n_points: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a DOS curve onto a uniform energy grid."""
    energies = np.asarray(energies, dtype=np.float32)
    dos = np.asarray(dos, dtype=np.float32)
    target_energy = np.linspace(float(energies.min()), float(energies.max()), n_points, dtype=np.float32)
    target_dos = np.interp(target_energy, energies, dos).astype(np.float32)
    return target_energy, target_dos


def debye_temperature_from_dos(csv_path: str | Path) -> float:
    """Estimate a Debye temperature from the second moment of the DOS."""
    table = load_dos_table(csv_path)
    energies = table["E"].to_numpy(dtype=np.float64)
    dos = np.clip(table["Total_DOS"].to_numpy(dtype=np.float64), a_min=0.0, a_max=None)

    total_weight = np.trapezoid(dos, energies)
    if total_weight <= 0:
        return 0.0

    mean_e2 = np.trapezoid(dos * energies**2, energies) / total_weight
    debye_energy_meV = math.sqrt((5.0 / 3.0) * mean_e2)
    return float(debye_energy_meV / KB_MEV_PER_K)


def prepare_dos_input_vector(
    csv_path: str | Path,
    n_points: int = 256,
    base_transform: str = "log1p",
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """Load one DOS curve and convert it into a fixed-length input vector."""
    table = load_dos_table(csv_path)
    _, dos = resample_dos_curve(
        table["E"].to_numpy(dtype=np.float32),
        table["Total_DOS"].to_numpy(dtype=np.float32),
        n_points=n_points,
    )
    if base_transform == "log1p":
        dos = np.log1p(np.clip(dos, a_min=0.0, a_max=None))
    elif base_transform != "none":
        raise ValueError(f"Unsupported transform: {base_transform}")
    if mean is not None and std is not None:
        dos = (dos - mean) / np.maximum(std, 1e-6)
    return dos.astype(np.float32)


def compute_dos_stats(
    csv_paths,
    n_points: int = 256,
    base_transform: str = "log1p",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-bin mean and standard deviation for DOS normalization."""
    vectors = [prepare_dos_input_vector(path, n_points=n_points, base_transform=base_transform) for path in csv_paths]
    matrix = np.stack(vectors).astype(np.float32)
    return matrix.mean(axis=0).astype(np.float32), matrix.std(axis=0).astype(np.float32)


def make_dos_tensor(
    csv_paths,
    n_points: int = 256,
    base_transform: str = "log1p",
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> torch.Tensor:
    """Stack many DOS vectors into a 2D PyTorch tensor."""
    vectors = [
        prepare_dos_input_vector(path, n_points=n_points, base_transform=base_transform, mean=mean, std=std)
        for path in csv_paths
    ]
    return torch.from_numpy(np.stack(vectors).astype(np.float32))


def pca_project(matrix: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Project a feature matrix onto its first principal components."""
    x = np.asarray(matrix, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    components = vt[:n_components]
    scores = x @ components.T
    return scores.astype(np.float32), components.astype(np.float32)
