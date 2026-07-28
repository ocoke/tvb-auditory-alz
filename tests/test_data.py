from __future__ import annotations

import hashlib
import io
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pytest

from rise_tvb379.data import (
    EDUCASE_COMMIT,
    EXPECTED_ANCHORS,
    PIPELINE_COMMIT,
    SOURCE_SPECS,
    DataValidationError,
    DownloadError,
    OfflineDataError,
    SourceSpec,
    build_b_by_severity,
    build_roi_definitions,
    download_sources,
    load_experiment_data,
    source_manifest_dataframe,
    transform_amyloid_to_b,
    validate_alignment,
)


def _aligned_labels() -> np.ndarray:
    labels = np.array(
        [f"L_region_{index}" for index in range(180)]
        + [f"R_region_{index}" for index in range(180)]
        + [f"Subcortical_{index}" for index in range(18)]
        + ["Brainstem"],
        dtype=str,
    )
    for label, index in EXPECTED_ANCHORS.items():
        labels[index] = label
    return labels


def _aligned_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.eye(379, dtype=float)
    weights[0, 1] = 0.25
    labels = _aligned_labels()
    amyloid = np.linspace(1.2, 3.7, 379)
    return weights, labels, amyloid


def test_pinned_source_specs_match_notebook() -> None:
    assert EDUCASE_COMMIT == "659d4fcbf58d74867fa9d10a874deac854532ee1"
    assert PIPELINE_COMMIT == "8be09e33e1131ed2f0764506940e6172de275285"
    assert set(SOURCE_SPECS) == {
        "structural_connectivity",
        "ad_left_cortex",
        "ad_right_cortex",
        "ad_subcortical",
        "region_labels",
    }
    assert {
        name: spec.sha256 for name, spec in SOURCE_SPECS.items()
    } == {
        "structural_connectivity": (
            "141fc993c84bde0b2f0ee0280ce1ccc47e1731ddcbf37845a4eef38dad9fa562"
        ),
        "ad_left_cortex": (
            "566e770e93f50d3378a0cf7d2dc8b1fa5af9475ca432463acf7bc2b5507907af"
        ),
        "ad_right_cortex": (
            "4f42c31c08e6d191d415953f763889f816d9c2443d115612f0c076cfd5b2f129"
        ),
        "ad_subcortical": (
            "f28926c7955db2c2762b5ba032f0bbde770a0da3d3705dd037a746382506d824"
        ),
        "region_labels": (
            "f9688592130a034210b482a0556fdc14383eaaf578bef39d2ddba072537e3484"
        ),
    }


def test_download_sources_uses_verified_offline_cache_atomically(
    tmp_path: Path,
) -> None:
    content = b"verified source bytes\n"
    digest = hashlib.sha256(content).hexdigest()
    spec = SourceSpec("source.txt", "https://example.test/source.txt", digest)
    cache_dir = tmp_path / "cache"
    data_dir = tmp_path / "data"
    cache_dir.mkdir()
    (cache_dir / spec.filename).write_bytes(content)

    records = download_sources(
        data_dir,
        cache_dir=cache_dir,
        offline=True,
        source_specs={"test_source": spec},
    )

    assert (data_dir / spec.filename).read_bytes() == content
    assert not list(data_dir.glob("*.tmp"))
    assert records[0].retrieved_from == "verified local validation cache"
    frame = source_manifest_dataframe(records)
    assert frame.columns.tolist() == [
        "source",
        "path",
        "sha256",
        "retrieved_from",
    ]


def test_offline_mode_never_calls_network(tmp_path: Path) -> None:
    content = b"missing"
    spec = SourceSpec(
        "missing.txt",
        "https://example.test/missing.txt",
        hashlib.sha256(content).hexdigest(),
    )
    called = False

    def forbidden_urlopen(*args: object, **kwargs: object) -> io.BytesIO:
        nonlocal called
        called = True
        return io.BytesIO(content)

    with pytest.raises(OfflineDataError):
        download_sources(
            tmp_path,
            offline=True,
            source_specs={"missing": spec},
            urlopen=forbidden_urlopen,
        )
    assert not called


def test_download_retries_and_preserves_destination_until_verified(
    tmp_path: Path,
) -> None:
    content = b"eventually valid"
    spec = SourceSpec(
        "source.txt",
        "https://example.test/source.txt",
        hashlib.sha256(content).hexdigest(),
    )
    destination = tmp_path / spec.filename
    destination.write_bytes(b"old invalid content")
    attempts = 0

    def flaky_urlopen(*args: object, **kwargs: object) -> io.BytesIO:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise URLError("temporary failure")
        return io.BytesIO(content)

    records = download_sources(
        tmp_path,
        retries=2,
        retry_backoff_seconds=0,
        source_specs={"test": spec},
        urlopen=flaky_urlopen,
    )
    assert attempts == 2
    assert destination.read_bytes() == content
    assert records[0].retrieved_from == spec.url


def test_failed_download_does_not_replace_existing_file(tmp_path: Path) -> None:
    expected = b"expected"
    spec = SourceSpec(
        "source.txt",
        "https://example.test/source.txt",
        hashlib.sha256(expected).hexdigest(),
    )
    destination = tmp_path / spec.filename
    destination.write_bytes(b"existing invalid file")

    with pytest.raises(DownloadError):
        download_sources(
            tmp_path,
            retries=1,
            retry_backoff_seconds=0,
            source_specs={"test": spec},
            urlopen=lambda *args, **kwargs: io.BytesIO(b"wrong download"),
        )
    assert destination.read_bytes() == b"existing invalid file"
    assert not list(tmp_path.glob("*.tmp"))


def test_alignment_and_roi_validation() -> None:
    weights, labels, amyloid = _aligned_arrays()
    validation = validate_alignment(weights, labels, amyloid)
    assert validation.label_to_index["L_A1"] == 23
    assert validation.relative_asymmetry > 0
    assert validation.data_quality_df.iloc[:9]["result"].all()

    rois = build_roi_definitions(validation.label_to_index)
    assert rois.a1_indices.tolist() == [23, 203]
    assert rois.music_indices.tolist() == [43, 223, 39, 219]
    assert rois.speech_indices.tolist() == [128, 308, 73, 253]
    assert len(rois.definition_df) == 10
    assert not rois.all_declared_indices.flags.writeable


def test_alignment_rejects_anchor_and_shape_errors() -> None:
    weights, labels, amyloid = _aligned_arrays()
    swapped = labels.copy()
    swapped[23], swapped[24] = swapped[24], swapped[23]
    with pytest.raises(DataValidationError, match="anchor"):
        validate_alignment(weights, swapped, amyloid)
    with pytest.raises(DataValidationError, match="379 amyloid values"):
        validate_alignment(weights, labels, amyloid[:-1])


def test_amyloid_transform_and_severity_vectors_match_notebook_formula() -> None:
    amyloid = np.array([1.4, 2.025, 2.65])
    transformed = transform_amyloid_to_b(amyloid)
    x0 = (2.65 - 1.4) / 2.0 + 1.4
    k = np.log(0.05 / ((0.02 + 0.001) - 0.02) - 1.0) / (2.65 - x0)
    expected = 0.05 / (1.0 + np.exp(k * (amyloid - x0))) + 0.02
    np.testing.assert_allclose(transformed, expected)

    full_amyloid = np.linspace(1.2, 3.7, 379)
    b_by_severity = build_b_by_severity(full_amyloid)
    np.testing.assert_allclose(b_by_severity[0.0], 0.07)
    np.testing.assert_allclose(
        b_by_severity[0.5],
        0.07 + 0.5 * (b_by_severity[1.0] - 0.07),
    )
    assert not b_by_severity[1.0].flags.writeable


def test_load_experiment_data_returns_pipeline_tables(tmp_path: Path) -> None:
    weights, labels, amyloid = _aligned_arrays()
    filenames = {
        "structural_connectivity": "weights.txt",
        "ad_left_cortex": "left.txt",
        "ad_right_cortex": "right.txt",
        "ad_subcortical": "subcortical.txt",
        "region_labels": "labels.txt",
    }
    placeholder_digest = hashlib.sha256(b"placeholder").hexdigest()
    specs = {
        key: SourceSpec(
            filename,
            f"https://example.test/{filename}",
            placeholder_digest,
        )
        for key, filename in filenames.items()
    }
    np.savetxt(tmp_path / filenames["structural_connectivity"], weights)
    np.savetxt(tmp_path / filenames["ad_left_cortex"], amyloid[:180])
    np.savetxt(tmp_path / filenames["ad_right_cortex"], amyloid[180:360])
    np.savetxt(tmp_path / filenames["ad_subcortical"], amyloid[360:])
    (tmp_path / filenames["region_labels"]).write_text(
        "\n".join(labels) + "\n"
    )

    loaded = load_experiment_data(tmp_path, source_specs=specs)

    assert loaded.weights.shape == (379, 379)
    assert loaded.ad_amyloid.shape == (379,)
    assert loaded.label_to_index["R_A1"] == 203
    assert loaded.baseline_b.shape == (379,)
    assert loaded.high_b.shape == (379,)
    assert loaded.roi_definition_df.shape[0] == 10
    assert loaded.roi_pathology_df.shape[0] == 10
    assert loaded.pathology_summary_df["severity"].tolist() == [0.0, 0.5, 1.0]
    assert not loaded.weights.flags.writeable
    assert not loaded.labels.flags.writeable
