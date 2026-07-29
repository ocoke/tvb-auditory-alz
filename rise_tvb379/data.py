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

from .config import (
    DT_CHECK_NETWORKS,
    N_REGIONS,
    SEVERITY_LABELS,
)

EDUCASE_COMMIT = "659d4fcbf58d74867fa9d10a874deac854532ee1"
PIPELINE_COMMIT = "8be09e33e1131ed2f0764506940e6172de275285"

def bilateral(*parcel_names: str) -> tuple[str, ...]:
    """Convert HCP parcel names into alternating left/right labels."""

    return tuple(
        label
        for parcel in parcel_names
        for label in (f"L_{parcel}", f"R_{parcel}")
    )


def ordered_union(*groups: Iterable[str]) -> tuple[str, ...]:
    """Combine label groups while preserving order and removing duplicates."""

    return tuple(
        dict.fromkeys(label for group in groups for label in group)
    )


ROI_GROUPS: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        "a1_input": {
            "labels": bilateral("A1"),
            "analysis_role": "stimulus",
            "interpretation": (
                "Bilateral primary auditory cortex receiving the external input."
            ),
        },
        "shared_early_auditory_relay": {
            "labels": bilateral("52", "MBelt", "LBelt", "RI"),
            "analysis_role": "shared_relay",
            "interpretation": (
                "Early auditory belt and retroinsular relay shared by both "
                "branches."
            ),
        },
        "shared_parabelt": {
            "labels": bilateral("PBelt"),
            "analysis_role": "shared_relay",
            "interpretation": (
                "Available HCP parabelt parcel. HCP-MMP does not separately "
                "represent rostral and caudal parabelt."
            ),
        },
        "shared_auditory_association": {
            "labels": bilateral("A4", "A5"),
            "analysis_role": "shared_relay",
            "interpretation": (
                "Nonprimary auditory association cortex downstream of parabelt."
            ),
        },
        "diagram_music_temporal_proxy": {
            "labels": bilateral("TA2", "STGa"),
            "analysis_role": "primary_music",
            "interpretation": (
                "Planum-polare/anterior auditory proxy from the pathway diagram. "
                "This is not parcel-level proof of music selectivity."
            ),
        },
        "music_memory_core_proxy": {
            "labels": bilateral("6ma", "24dd"),
            "analysis_role": "primary_music",
            "interpretation": (
                "Parcel approximations of ventral pre-SMA and caudal anterior "
                "cingulate musical-memory regions."
            ),
        },
        "music_semantic_task_associated": {
            "labels": (
                "R_9m",
                "L_25",
                "R_TE1a",
                "L_TF",
                "L_STSda",
            ),
            "analysis_role": "secondary_music_memory",
            "interpretation": (
                "HCP-MMP parcels mapped from cortical peaks in the Platel et al. "
                "semantic-greater-than-episodic musical-memory contrast."
            ),
        },
        "music_episodic_task_associated": {
            "labels": (
                "R_IP1",
                "R_PCV",
                "R_11l",
                "R_8Av",
            ),
            "analysis_role": "secondary_music_memory",
            "interpretation": (
                "HCP-MMP parcels mapped from cortical peaks in the Platel et al. "
                "episodic-greater-than-semantic musical-memory contrast."
            ),
        },
        "music_anterior_temporal_context": {
            "labels": bilateral("TGd", "TGv", "TE1a", "TE2a"),
            "analysis_role": "diagnostic_only",
            "interpretation": (
                "Anterior temporal semantic and conceptual context."
            ),
        },
        "music_frontoparietal_context": {
            "labels": bilateral(
                "10r",
                "47l",
                "a9-46v",
                "46",
                "8Ad",
                "8Av",
                "PFm",
                "PGi",
                "PGs",
            ),
            "analysis_role": "diagnostic_only",
            "interpretation": (
                "Diagram-mapped ventrolateral/dorsolateral prefrontal and "
                "inferior-parietal context."
            ),
        },
        "music_medial_temporal_context": {
            "labels": (
                "Left-Hippocampus",
                "Right-Hippocampus",
                "Left-Amygdala",
                "Right-Amygdala",
            ),
            "analysis_role": "diagnostic_only",
            "interpretation": (
                "Medial-temporal context from the diagram. The Jansen-Rit "
                "model does not implement memory encoding or retrieval."
            ),
        },
        "speech_posterior_temporal": {
            "labels": bilateral("PSL", "PFcm", "STSdp", "TPOJ1"),
            "analysis_role": "primary_speech",
            "interpretation": (
                "Bilateral posterior temporal and perisylvian "
                "auditory-language targets."
            ),
        },
        "speech_left_frontal": {
            "labels": (
                "L_44",
                "L_45",
                "L_47l",
                "L_FOP4",
                "L_8C",
                "L_SCEF",
            ),
            "analysis_role": "primary_speech",
            "interpretation": (
                "Left-lateralized inferior and medial frontal language targets."
            ),
        },
        "speech_dorsal_left": {
            "labels": (
                "L_PSL",
                "L_PFcm",
                "L_TPOJ1",
                "L_PF",
                "L_PFm",
                "L_PGi",
                "L_PGs",
                "L_PGp",
                "L_55b",
                "L_6r",
                "L_4",
                "L_44",
                "L_FOP4",
                "L_8C",
                "L_SCEF",
            ),
            "analysis_role": "diagnostic_only",
            "interpretation": (
                "Left temporoparietal-to-premotor/articulatory dorsal stream."
            ),
        },
        "speech_ventral": {
            "labels": ordered_union(
                bilateral(
                    "STSda",
                    "STSdp",
                    "STSva",
                    "STSvp",
                    "TE1a",
                    "TE1m",
                    "TE1p",
                    "TE2a",
                    "TE2p",
                    "TGd",
                    "TGv",
                ),
                ("L_45", "L_47l"),
            ),
            "analysis_role": "diagnostic_only",
            "interpretation": (
                "Bilateral temporal semantic stream with left "
                "inferior-frontal outputs."
            ),
        },
    }
)

A1_LABELS = tuple(ROI_GROUPS["a1_input"]["labels"])
MUSIC_LABELS = ordered_union(
    *(
        tuple(specification["labels"])
        for specification in ROI_GROUPS.values()
        if specification["analysis_role"] == "primary_music"
    )
)
SPEECH_LABELS = ordered_union(
    *(
        tuple(specification["labels"])
        for specification in ROI_GROUPS.values()
        if specification["analysis_role"] == "primary_speech"
    )
)
SEMANTIC_MEMORY_LABELS = tuple(
    ROI_GROUPS["music_semantic_task_associated"]["labels"]
)
EPISODIC_MEMORY_LABELS = tuple(
    ROI_GROUPS["music_episodic_task_associated"]["labels"]
)
ALL_ROI_LABELS = ordered_union(
    *(tuple(specification["labels"]) for specification in ROI_GROUPS.values())
)

NETWORK_LABELS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "music": MUSIC_LABELS,
        "speech": SPEECH_LABELS,
        "shared_auditory_relay": ordered_union(
            tuple(ROI_GROUPS["shared_early_auditory_relay"]["labels"]),
            tuple(ROI_GROUPS["shared_parabelt"]["labels"]),
            tuple(ROI_GROUPS["shared_auditory_association"]["labels"]),
        ),
        **{
            group_name: tuple(specification["labels"])
            for group_name, specification in ROI_GROUPS.items()
            if group_name != "a1_input"
        },
    }
)

if not set(DT_CHECK_NETWORKS).issubset(NETWORK_LABELS):
    raise RuntimeError("Inferential convergence networks are not defined.")

EXPECTED_ANCHORS: Mapping[str, int] = MappingProxyType(
    {
        "L_A1": 23,
        "R_A1": 203,
        "L_TA2": 106,
        "R_TA2": 286,
        "L_STGa": 122,
        "R_STGa": 302,
        "L_6ma": 43,
        "R_6ma": 223,
        "L_24dd": 39,
        "R_24dd": 219,
        "L_PSL": 24,
        "R_PSL": 204,
        "L_PFcm": 104,
        "R_PFcm": 284,
        "L_STSdp": 128,
        "R_STSdp": 308,
        "L_TPOJ1": 138,
        "R_TPOJ1": 318,
        "L_44": 73,
        "L_45": 74,
        "L_47l": 75,
        "L_FOP4": 107,
        "L_8C": 72,
        "L_SCEF": 42,
        "R_9m": 248,
        "L_25": 163,
        "R_TE1a": 311,
        "L_TF": 134,
        "L_STSda": 127,
        "R_IP1": 324,
        "R_PCV": 206,
        "R_11l": 270,
        "R_8Av": 246,
    }
)

MUSIC_MEMORY_PEAK_MAPPINGS: tuple[dict[str, object], ...] = (
    {
        "source_contrast": "semantic > episodic",
        "reported_region": "bilateral medial frontal cortex (BA 11/10)",
        "spm99_x": 0,
        "spm99_y": 60,
        "spm99_z": 10,
        "hcp_label": "R_9m",
        "mapping_distance_mm": 1.0,
        "mapping_note": "nearest volumetric HCP-MMP parcel",
    },
    {
        "source_contrast": "semantic > episodic",
        "reported_region": "bilateral medial frontal cortex (BA 11/10)",
        "spm99_x": -4,
        "spm99_y": 18,
        "spm99_z": -18,
        "hcp_label": "L_25",
        "mapping_distance_mm": 0.0,
        "mapping_note": "coordinate falls inside parcel",
    },
    {
        "source_contrast": "semantic > episodic",
        "reported_region": "right middle temporal gyrus (BA 21)",
        "spm99_x": 56,
        "spm99_y": 4,
        "spm99_z": -24,
        "hcp_label": "R_TE1a",
        "mapping_distance_mm": 0.0,
        "mapping_note": "coordinate falls inside parcel",
    },
    {
        "source_contrast": "semantic > episodic",
        "reported_region": "left inferior/middle temporal gyri (BA 20/21)",
        "spm99_x": -48,
        "spm99_y": -26,
        "spm99_z": -22,
        "hcp_label": "L_TF",
        "mapping_distance_mm": 0.0,
        "mapping_note": "coordinate falls inside parcel",
    },
    {
        "source_contrast": "semantic > episodic",
        "reported_region": "left inferior/middle temporal gyri (BA 20/21)",
        "spm99_x": -54,
        "spm99_y": -2,
        "spm99_z": -18,
        "hcp_label": "L_STSda",
        "mapping_distance_mm": 0.0,
        "mapping_note": "coordinate falls inside parcel",
    },
    {
        "source_contrast": "episodic > semantic",
        "reported_region": "right precuneus/parietal cortex (BA 7/19)",
        "spm99_x": 36,
        "spm99_y": -66,
        "spm99_z": 38,
        "hcp_label": "R_IP1",
        "mapping_distance_mm": 2.0,
        "mapping_note": "nearest volumetric HCP-MMP parcel",
    },
    {
        "source_contrast": "episodic > semantic",
        "reported_region": "precuneus (BA 7)",
        "spm99_x": 4,
        "spm99_y": -56,
        "spm99_z": 42,
        "hcp_label": "R_PCV",
        "mapping_distance_mm": 0.0,
        "mapping_note": "coordinate falls inside parcel",
    },
    {
        "source_contrast": "episodic > semantic",
        "reported_region": "right superior frontal gyrus (BA 11)",
        "spm99_x": 34,
        "spm99_y": 52,
        "spm99_z": -14,
        "hcp_label": "R_11l",
        "mapping_distance_mm": 1.0,
        "mapping_note": (
            "nearest parcel consistent with reported BA11 anatomy; "
            "volumetric boundary"
        ),
    },
    {
        "source_contrast": "episodic > semantic",
        "reported_region": "right middle frontal gyrus (BA 8/9)",
        "spm99_x": 38,
        "spm99_y": 12,
        "spm99_z": 44,
        "hcp_label": "R_8Av",
        "mapping_distance_mm": 1.0,
        "mapping_note": "nearest volumetric HCP-MMP parcel",
    },
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
    semantic_memory_labels: tuple[str, ...]
    episodic_memory_labels: tuple[str, ...]
    a1_indices: np.ndarray
    music_indices: np.ndarray
    speech_indices: np.ndarray
    semantic_memory_indices: np.ndarray
    episodic_memory_indices: np.ndarray
    network_labels: Mapping[str, tuple[str, ...]]
    network_indices: Mapping[str, np.ndarray]
    all_declared_indices: np.ndarray
    primary_counterfactual_fixed_indices: np.ndarray
    memory_counterfactual_fixed_indices: np.ndarray
    definition_df: pd.DataFrame
    peak_mapping_df: pd.DataFrame


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

    missing = sorted(set(ALL_ROI_LABELS) - set(label_to_index))
    if missing:
        raise DataValidationError(
            "Missing predeclared parcel labels: " + ", ".join(missing)
        )

    for group_name, specification in ROI_GROUPS.items():
        group_labels = tuple(specification["labels"])
        if len(group_labels) != len(set(group_labels)):
            raise DataValidationError(
                f"Duplicate label inside ROI group {group_name!r}."
            )

    primary_overlap = sorted(set(MUSIC_LABELS) & set(SPEECH_LABELS))
    if primary_overlap:
        raise DataValidationError(
            f"Primary music and speech proxies overlap: {primary_overlap}"
        )
    secondary_overlap = sorted(
        set(SEMANTIC_MEMORY_LABELS) & set(EPISODIC_MEMORY_LABELS)
    )
    if secondary_overlap:
        raise DataValidationError(
            "Secondary musical-memory proxy sets overlap: "
            f"{secondary_overlap}"
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
    semantic_memory_indices = np.array(
        [label_to_index[label] for label in SEMANTIC_MEMORY_LABELS],
        dtype=int,
    )
    episodic_memory_indices = np.array(
        [label_to_index[label] for label in EPISODIC_MEMORY_LABELS],
        dtype=int,
    )
    network_indices = {
        network_name: np.array(
            [label_to_index[label] for label in network_labels],
            dtype=int,
        )
        for network_name, network_labels in NETWORK_LABELS.items()
    }
    all_declared_indices = np.array(
        sorted(label_to_index[label] for label in ALL_ROI_LABELS),
        dtype=int,
    )
    primary_counterfactual_fixed_indices = np.array(
        sorted(
            set(a1_indices.tolist())
            | set(music_indices.tolist())
            | set(speech_indices.tolist())
        ),
        dtype=int,
    )
    memory_counterfactual_fixed_indices = np.array(
        sorted(
            set(a1_indices.tolist())
            | set(semantic_memory_indices.tolist())
            | set(episodic_memory_indices.tolist())
        ),
        dtype=int,
    )

    rows: list[dict[str, object]] = []
    for group_name, specification in ROI_GROUPS.items():
        for label in tuple(specification["labels"]):
            rows.append(
                {
                    "network": group_name,
                    "analysis_role": specification["analysis_role"],
                    "label": label,
                    "zero_based_index": label_to_index[label],
                    "interpretation": specification["interpretation"],
                }
            )

    peak_mapping_df = pd.DataFrame(MUSIC_MEMORY_PEAK_MAPPINGS)
    actual_semantic_mapping = set(
        peak_mapping_df.loc[
            peak_mapping_df["source_contrast"] == "semantic > episodic",
            "hcp_label",
        ]
    )
    actual_episodic_mapping = set(
        peak_mapping_df.loc[
            peak_mapping_df["source_contrast"] == "episodic > semantic",
            "hcp_label",
        ]
    )
    if actual_semantic_mapping != set(SEMANTIC_MEMORY_LABELS):
        raise DataValidationError(
            "Semantic peak mapping and ROI labels disagree."
        )
    if actual_episodic_mapping != set(EPISODIC_MEMORY_LABELS):
        raise DataValidationError(
            "Episodic peak mapping and ROI labels disagree."
        )

    for array in (
        a1_indices,
        music_indices,
        speech_indices,
        semantic_memory_indices,
        episodic_memory_indices,
        all_declared_indices,
        primary_counterfactual_fixed_indices,
        memory_counterfactual_fixed_indices,
        *network_indices.values(),
    ):
        array.setflags(write=False)
    return ROISelections(
        a1_labels=A1_LABELS,
        music_labels=MUSIC_LABELS,
        speech_labels=SPEECH_LABELS,
        semantic_memory_labels=SEMANTIC_MEMORY_LABELS,
        episodic_memory_labels=EPISODIC_MEMORY_LABELS,
        a1_indices=a1_indices,
        music_indices=music_indices,
        speech_indices=speech_indices,
        semantic_memory_indices=semantic_memory_indices,
        episodic_memory_indices=episodic_memory_indices,
        network_labels=NETWORK_LABELS,
        network_indices=MappingProxyType(network_indices),
        all_declared_indices=all_declared_indices,
        primary_counterfactual_fixed_indices=(
            primary_counterfactual_fixed_indices
        ),
        memory_counterfactual_fixed_indices=(
            memory_counterfactual_fixed_indices
        ),
        definition_df=pd.DataFrame(rows),
        peak_mapping_df=peak_mapping_df,
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
