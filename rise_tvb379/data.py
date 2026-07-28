"""Verified source acquisition and data preparation for the TVB experiment."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .config import N_REGIONS, SEVERITY_LABELS

EDUCASE_COMMIT = "659d4fcbf58d74867fa9d10a874deac854532ee1"
PIPELINE_COMMIT = "8be09e33e1131ed2f0764506940e6172de275285"

A1_LABELS = ("L_A1", "R_A1")
MUSIC_LABELS = ("L_6ma", "R_6ma", "L_24dd", "R_24dd")
SPEECH_LABELS = ("L_STSdp", "R_STSdp", "L_44", "R_44")

EXPECTED_ANCHORS: Mapping[str, int] = MappingProxyType(
    {
        "L_A1": 23,
        "R_A1": 203,
        "L_6ma": 43,
        "R_6ma": 223,
        "L_24dd": 39,
        "R_24dd": 219,
        "L_STSdp": 128,
        "R_STSdp": 308,
        "L_44": 73,
        "R_44": 253,
    }
)


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """A content-addressed source file."""

    filename: str
    url: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.filename or Path(self.filename).name != self.filename:
            raise ValueError("Source filename must be a simple file name.")
        if not self.url.startswith(("https://", "http://")):
            raise ValueError("Source URL must use HTTP or HTTPS.")
        if len(self.sha256) != 64:
            raise ValueError("Source SHA-256 must contain 64 hexadecimal characters.")
        try:
            int(self.sha256, 16)
        except ValueError as error:
            raise ValueError("Source SHA-256 is not hexadecimal.") from error


SOURCE_SPECS: Mapping[str, SourceSpec] = MappingProxyType(
    {
        "structural_connectivity": SourceSpec(
            filename="avg_healthy_normSC_mod.txt",
            url=(
                "https://raw.githubusercontent.com/BrainModes/"
                "TVB_EducaseAD_molecular_pathways_TVB/"
                f"{EDUCASE_COMMIT}/avg_healthy_normSC_mod.txt"
            ),
            sha256=(
                "141fc993c84bde0b2f0ee0280ce1ccc47e1731ddcbf37845a4eef38dad9fa562"
            ),
        ),
        "ad_left_cortex": SourceSpec(
            filename="AD_LH.txt",
            url=(
                "https://raw.githubusercontent.com/BrainModes/"
                "TVB_EducaseAD_molecular_pathways_TVB/"
                f"{EDUCASE_COMMIT}/_AD/AD_LH.txt"
            ),
            sha256=(
                "566e770e93f50d3378a0cf7d2dc8b1fa5af9475ca432463acf7bc2b5507907af"
            ),
        ),
        "ad_right_cortex": SourceSpec(
            filename="AD_RH.txt",
            url=(
                "https://raw.githubusercontent.com/BrainModes/"
                "TVB_EducaseAD_molecular_pathways_TVB/"
                f"{EDUCASE_COMMIT}/_AD/AD_RH.txt"
            ),
            sha256=(
                "4f42c31c08e6d191d415953f763889f816d9c2443d115612f0c076cfd5b2f129"
            ),
        ),
        "ad_subcortical": SourceSpec(
            filename="AD_subcortical.txt",
            url=(
                "https://raw.githubusercontent.com/BrainModes/"
                "TVB_EducaseAD_molecular_pathways_TVB/"
                f"{EDUCASE_COMMIT}/_AD/AD_subcortical.txt"
            ),
            sha256=(
                "f28926c7955db2c2762b5ba032f0bbde770a0da3d3705dd037a746382506d824"
            ),
        ),
        "region_labels": SourceSpec(
            filename="region_labels.txt",
            url=(
                "https://raw.githubusercontent.com/BrainModes/"
                f"ADNI-TVB-pipeline/{PIPELINE_COMMIT}/misc_files/region_labels.txt"
            ),
            sha256=(
                "f9688592130a034210b482a0556fdc14383eaaf578bef39d2ddba072537e3484"
            ),
        ),
    }
)


class DataError(RuntimeError):
    """Base class for experiment data failures."""


class ChecksumMismatchError(DataError):
    """A source did not match its pinned checksum."""


class DownloadError(DataError):
    """A verified source could not be downloaded."""


class OfflineDataError(DataError):
    """A required verified source is unavailable in offline mode."""


class DataValidationError(DataError):
    """Loaded experiment arrays are not correctly aligned."""


@dataclass(frozen=True, slots=True)
class SourceManifestRecord:
    source: str
    path: str
    sha256: str
    retrieved_from: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "path": self.path,
            "sha256": self.sha256,
            "retrieved_from": self.retrieved_from,
        }


@dataclass(frozen=True, slots=True)
class AlignmentValidation:
    label_to_index: Mapping[str, int]
    relative_asymmetry: float
    data_quality_df: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ROISelections:
    a1_labels: tuple[str, ...]
    music_labels: tuple[str, ...]
    speech_labels: tuple[str, ...]
    a1_indices: np.ndarray
    music_indices: np.ndarray
    speech_indices: np.ndarray
    all_declared_indices: np.ndarray
    definition_df: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ExperimentData:
    """Validated arrays and derived tables consumed by the simulation pipeline."""

    weights: np.ndarray
    labels: np.ndarray
    ad_amyloid: np.ndarray
    label_to_index: Mapping[str, int]
    relative_asymmetry: float
    rois: ROISelections
    baseline_b: np.ndarray
    high_b: np.ndarray
    b_by_severity: Mapping[float, np.ndarray]
    data_quality_df: pd.DataFrame
    roi_definition_df: pd.DataFrame
    roi_pathology_df: pd.DataFrame
    pathology_summary_df: pd.DataFrame


UrlOpen = Callable[..., BinaryIO]


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest_dataframe(
    records: Iterable[SourceManifestRecord],
) -> pd.DataFrame:
    """Convert source records to the notebook-compatible manifest schema."""

    return pd.DataFrame(
        [record.to_dict() for record in records],
        columns=("source", "path", "sha256", "retrieved_from"),
    )


def data_cache_from_environment(
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve the optional ``RISE_DATA_CACHE`` directory."""

    environment = os.environ if environ is None else environ
    value = environment.get("RISE_DATA_CACHE")
    return Path(value).expanduser() if value else None


def _stream_to_atomic_verified_file(
    stream: BinaryIO,
    destination: Path,
    expected_sha256: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        with os.fdopen(file_descriptor, "wb") as output:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ChecksumMismatchError(
                "SHA-256 mismatch for "
                f"{destination.name}: expected {expected_sha256}, "
                f"got {actual_sha256}."
            )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _copy_verified_cache(
    cache_path: Path,
    destination: Path,
    expected_sha256: str,
) -> bool:
    if not cache_path.is_file() or sha256_file(cache_path) != expected_sha256:
        return False
    with cache_path.open("rb") as stream:
        _stream_to_atomic_verified_file(stream, destination, expected_sha256)
    return True


def _download_verified(
    spec: SourceSpec,
    destination: Path,
    *,
    retries: int,
    timeout_seconds: float,
    retry_backoff_seconds: float,
    urlopen: UrlOpen,
) -> None:
    last_error: BaseException | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                spec.url,
                headers={"User-Agent": "rise-tvb379/1"},
            )
            with urlopen(request, timeout=timeout_seconds) as response:
                _stream_to_atomic_verified_file(
                    response, destination, spec.sha256
                )
            return
        except (
            ChecksumMismatchError,
            OSError,
            TimeoutError,
            urllib.error.URLError,
        ) as error:
            last_error = error
            if attempt + 1 < retries and retry_backoff_seconds:
                time.sleep(retry_backoff_seconds * (2**attempt))
    raise DownloadError(
        f"Could not download and verify {spec.filename} after {retries} attempts."
    ) from last_error


def download_sources(
    data_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    retries: int = 3,
    timeout_seconds: float = 120.0,
    retry_backoff_seconds: float = 0.5,
    source_specs: Mapping[str, SourceSpec] = SOURCE_SPECS,
    urlopen: UrlOpen = urllib.request.urlopen,
) -> tuple[SourceManifestRecord, ...]:
    """Materialize all pinned inputs without exposing partial downloads.

    Existing destination files and optional cache entries are only accepted
    after checksum verification. In offline mode the function never invokes
    ``urlopen``.
    """

    if retries < 1:
        raise ValueError("retries must be at least 1.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds cannot be negative.")

    destination_dir = Path(data_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_dir) if cache_dir is not None else None
    records: list[SourceManifestRecord] = []

    for source_name, spec in source_specs.items():
        destination = destination_dir / spec.filename
        retrieved_from: str
        if (
            destination.is_file()
            and sha256_file(destination) == spec.sha256
        ):
            retrieved_from = "existing verified file"
        else:
            candidate = cache_path / spec.filename if cache_path else None
            if candidate is not None and _copy_verified_cache(
                candidate, destination, spec.sha256
            ):
                retrieved_from = "verified local validation cache"
            elif offline:
                raise OfflineDataError(
                    f"Verified source {source_name!r} is unavailable offline: "
                    f"{spec.filename}."
                )
            else:
                _download_verified(
                    spec,
                    destination,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                    retry_backoff_seconds=retry_backoff_seconds,
                    urlopen=urlopen,
                )
                retrieved_from = spec.url

        actual_sha256 = sha256_file(destination)
        if actual_sha256 != spec.sha256:
            raise ChecksumMismatchError(
                f"SHA-256 mismatch for {source_name}: expected {spec.sha256}, "
                f"got {actual_sha256}."
            )
        records.append(
            SourceManifestRecord(
                source=source_name,
                path=str(destination.resolve()),
                sha256=actual_sha256,
                retrieved_from=retrieved_from,
            )
        )
    return tuple(records)


def validate_alignment(
    weights: np.ndarray,
    labels: np.ndarray,
    ad_amyloid: np.ndarray,
    *,
    n_regions: int = N_REGIONS,
    expected_anchors: Mapping[str, int] = EXPECTED_ANCHORS,
) -> AlignmentValidation:
    """Validate the exact matrix/vector/label alignment used by the model."""

    weights = np.asarray(weights)
    labels = np.asarray(labels, dtype=str)
    ad_amyloid = np.asarray(ad_amyloid)
    label_values = labels.tolist() if labels.ndim == 1 else []

    hard_checks = {
        "SC is 379 x 379": weights.shape == (n_regions, n_regions),
        "379 unique labels": labels.shape == (n_regions,)
        and len(set(label_values)) == n_regions,
        "379 amyloid values": ad_amyloid.shape == (n_regions,),
        "SC values finite": bool(np.isfinite(weights).all()),
        "SC values nonnegative": bool((weights >= 0).all()),
        "amyloid values finite": bool(np.isfinite(ad_amyloid).all()),
        "left cortex occupies 0:180": labels.shape == (n_regions,)
        and all(label.startswith("L_") for label in label_values[:180]),
        "right cortex occupies 180:360": labels.shape == (n_regions,)
        and all(label.startswith("R_") for label in label_values[180:360]),
        "brainstem is final parcel": labels.shape == (n_regions,)
        and label_values[-1] == "Brainstem",
    }
    failed_checks = [name for name, passed in hard_checks.items() if not passed]
    if failed_checks:
        raise DataValidationError(
            "Input validation failed: " + "; ".join(failed_checks)
        )

    label_to_index = {
        label: index for index, label in enumerate(label_values)
    }
    anchor_failures = {
        label: (label_to_index.get(label), expected_index)
        for label, expected_index in expected_anchors.items()
        if label_to_index.get(label) != expected_index
    }
    if anchor_failures:
        raise DataValidationError(
            f"Parcel-order anchor check failed: {anchor_failures}"
        )

    weights_norm = float(np.linalg.norm(weights))
    relative_asymmetry = (
        float(np.linalg.norm(weights - weights.T) / weights_norm)
        if weights_norm
        else 0.0
    )
    data_quality_df = pd.DataFrame(
        {
            "check": list(hard_checks)
            + [
                "SC minimum",
                "SC maximum",
                "SC relative asymmetry",
                "amyloid minimum",
                "amyloid maximum",
                "amyloid mean",
            ],
            "result": list(hard_checks.values())
            + [
                float(weights.min()),
                float(weights.max()),
                relative_asymmetry,
                float(ad_amyloid.min()),
                float(ad_amyloid.max()),
                float(ad_amyloid.mean()),
            ],
        }
    )
    return AlignmentValidation(
        label_to_index=MappingProxyType(label_to_index),
        relative_asymmetry=relative_asymmetry,
        data_quality_df=data_quality_df,
    )


def _readonly_array(values: np.ndarray, *, dtype: object | None = None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def build_roi_definitions(
    label_to_index: Mapping[str, int],
) -> ROISelections:
    """Resolve and validate the predeclared stimulation/target parcels."""

    missing = [
        label
        for label in (*A1_LABELS, *MUSIC_LABELS, *SPEECH_LABELS)
        if label not in label_to_index
    ]
    if missing:
        raise DataValidationError(
            "Missing predeclared parcel labels: " + ", ".join(missing)
        )

    a1_indices = np.array(
        [label_to_index[label] for label in A1_LABELS], dtype=int
    )
    music_indices = np.array(
        [label_to_index[label] for label in MUSIC_LABELS], dtype=int
    )
    speech_indices = np.array(
        [label_to_index[label] for label in SPEECH_LABELS], dtype=int
    )
    if len(music_indices) != len(speech_indices):
        raise DataValidationError(
            "Music and speech proxy groups must have equal size."
        )
    all_declared_indices = np.concatenate(
        (a1_indices, music_indices, speech_indices)
    )
    if len(np.unique(all_declared_indices)) != len(all_declared_indices):
        raise DataValidationError(
            "A parcel appears in more than one predeclared group."
        )

    rows: list[dict[str, object]] = []
    for network, labels, interpretation in (
        (
            "A1 seed",
            A1_LABELS,
            "bilateral primary auditory cortex stimulation",
        ),
        (
            "Music proxy",
            MUSIC_LABELS,
            "parcel approximations of ventral pre-SMA and "
            "caudal/anterior cingulate targets",
        ),
        (
            "Speech proxy",
            SPEECH_LABELS,
            "parcel approximations of posterior STS and "
            "inferior-frontal targets",
        ),
    ):
        for label in labels:
            rows.append(
                {
                    "network": network,
                    "label": label,
                    "zero_based_index": label_to_index[label],
                    "interpretation": interpretation,
                }
            )

    for array in (
        a1_indices,
        music_indices,
        speech_indices,
        all_declared_indices,
    ):
        array.setflags(write=False)
    return ROISelections(
        a1_labels=A1_LABELS,
        music_labels=MUSIC_LABELS,
        speech_labels=SPEECH_LABELS,
        a1_indices=a1_indices,
        music_indices=music_indices,
        speech_indices=speech_indices,
        all_declared_indices=all_declared_indices,
        definition_df=pd.DataFrame(rows),
    )


def transform_amyloid_to_b(
    amyloid: np.ndarray,
    *,
    max_val: float = 0.05,
    min_val: float = 0.02,
    amyloid_max: float = 2.65,
    amyloid_offset: float = 1.4,
) -> np.ndarray:
    """Apply the Stefanovski amyloid-to-inhibitory-rate transformation."""

    amyloid = np.asarray(amyloid, dtype=float)
    x0 = (amyloid_max - amyloid_offset) / 2.0 + amyloid_offset
    k = (
        np.log(max_val / ((min_val + 0.001) - min_val) - 1.0)
        / (amyloid_max - x0)
    )
    transformed = max_val / (1.0 + np.exp(k * (amyloid - x0))) + min_val
    if not np.isfinite(transformed).all():
        raise DataValidationError(
            "The transformed inhibitory vector contains nonfinite values."
        )
    if transformed.size and (
        float(transformed.min()) < 0.0199
        or float(transformed.max()) > 0.0701
    ):
        raise DataValidationError(
            "The transformed inhibitory vector is outside the expected range."
        )
    return transformed


def build_b_by_severity(
    ad_amyloid: np.ndarray,
    *,
    n_regions: int = N_REGIONS,
    baseline_value: float = 0.07,
) -> Mapping[float, np.ndarray]:
    """Build baseline, intermediate, and high regional ``b`` vectors."""

    amyloid = np.asarray(ad_amyloid, dtype=float)
    if amyloid.shape != (n_regions,):
        raise DataValidationError(
            f"ad_amyloid must be a {n_regions}-element vector."
        )
    baseline = np.full(n_regions, baseline_value, dtype=float)
    high = transform_amyloid_to_b(amyloid)
    values: dict[float, np.ndarray] = {}
    for severity in SEVERITY_LABELS:
        vector = baseline + severity * (high - baseline)
        vector.setflags(write=False)
        values[severity] = vector
    return MappingProxyType(values)


def _pathology_tables(
    ad_amyloid: np.ndarray,
    rois: ROISelections,
    b_by_severity: Mapping[float, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pathology_summary_df = pd.DataFrame(
        [
            {
                "severity": severity,
                "condition": SEVERITY_LABELS[severity],
                "b_min": float(values.min()),
                "b_mean": float(values.mean()),
                "b_max": float(values.max()),
                "mean_tau_i_ms": float(np.mean(1.0 / values)),
            }
            for severity, values in b_by_severity.items()
        ]
    )
    roi_pathology_df = rois.definition_df.copy()
    indices = roi_pathology_df["zero_based_index"].to_numpy(dtype=int)
    high = b_by_severity[1.0]
    roi_pathology_df["surrogate_amyloid"] = ad_amyloid[indices]
    roi_pathology_df["baseline_b"] = b_by_severity[0.0][indices]
    roi_pathology_df["high_b"] = high[indices]
    roi_pathology_df["b_reduction"] = (
        roi_pathology_df["baseline_b"] - roi_pathology_df["high_b"]
    )
    return pathology_summary_df, roi_pathology_df


def load_experiment_data(
    data_dir: str | Path,
    *,
    source_specs: Mapping[str, SourceSpec] = SOURCE_SPECS,
) -> ExperimentData:
    """Load all source files, validate alignment, and derive model inputs."""

    directory = Path(data_dir)
    required_keys = {
        "structural_connectivity",
        "ad_left_cortex",
        "ad_right_cortex",
        "ad_subcortical",
        "region_labels",
    }
    missing_keys = sorted(required_keys.difference(source_specs))
    if missing_keys:
        raise DataValidationError(
            "Missing source specifications: " + ", ".join(missing_keys)
        )

    def source_path(name: str) -> Path:
        path = directory / source_specs[name].filename
        if not path.is_file():
            raise DataValidationError(f"Required source file is missing: {path}")
        return path

    weights = np.loadtxt(source_path("structural_connectivity"))
    labels = np.array(
        [
            line.strip()
            for line in source_path("region_labels").read_text().splitlines()
            if line.strip()
        ],
        dtype=str,
    )
    ad_amyloid = np.concatenate(
        [
            np.atleast_1d(np.loadtxt(source_path("ad_left_cortex"))),
            np.atleast_1d(np.loadtxt(source_path("ad_right_cortex"))),
            np.atleast_1d(np.loadtxt(source_path("ad_subcortical"))),
        ]
    )

    validation = validate_alignment(weights, labels, ad_amyloid)
    rois = build_roi_definitions(validation.label_to_index)
    b_by_severity = build_b_by_severity(ad_amyloid)
    pathology_summary_df, roi_pathology_df = _pathology_tables(
        ad_amyloid, rois, b_by_severity
    )

    weights = _readonly_array(weights, dtype=float)
    labels = _readonly_array(labels, dtype=str)
    ad_amyloid = _readonly_array(ad_amyloid, dtype=float)
    return ExperimentData(
        weights=weights,
        labels=labels,
        ad_amyloid=ad_amyloid,
        label_to_index=validation.label_to_index,
        relative_asymmetry=validation.relative_asymmetry,
        rois=rois,
        baseline_b=b_by_severity[0.0],
        high_b=b_by_severity[1.0],
        b_by_severity=b_by_severity,
        data_quality_df=validation.data_quality_df,
        roi_definition_df=rois.definition_df,
        roi_pathology_df=roi_pathology_df,
        pathology_summary_df=pathology_summary_df,
    )


load_and_validate_data = load_experiment_data
