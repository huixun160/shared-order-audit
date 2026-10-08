"""Adapter tests against real files.

These run only if VP001 has been downloaded and extracted, so a fresh clone still has a
green test suite:

    python scripts/01_download_data.py --dataset shin2018 --subjects 1
    pytest
"""

from __future__ import annotations

import numpy as np
import pytest

from eegfnirs.data import shin2018
from eegfnirs.utils import project_paths

PATHS = project_paths()
SUBJECT_DIR = PATHS.interim / "shin2018" / "VP001"

pytestmark = pytest.mark.skipif(
    not (SUBJECT_DIR / "eeg").is_dir(),
    reason="VP001 not extracted; run scripts/01_download_data.py --subjects 1",
)


def test_all_runs_present():
    for run in shin2018.ALL_RUNS:
        assert (SUBJECT_DIR / "eeg" / (run + ".vhdr")).exists(), run


def test_nirs_sessions_map_by_inf_notes():
    mapping = shin2018.nirs_session_map(SUBJECT_DIR)
    assert set(mapping) == set(shin2018.ALL_RUNS)
    assert len(set(mapping.values())) == len(shin2018.ALL_RUNS)  # no session reused


@pytest.mark.parametrize("run", shin2018.ALL_RUNS)
def test_blocks_segment_and_validate(run):
    """The trigger mapping was reverse-engineered; this is the guard that it holds."""
    task = shin2018.task_of_run(run)
    markers = shin2018.parse_vmrk(SUBJECT_DIR / "eeg" / (run + ".vmrk"))
    blocks = shin2018.segment_blocks(markers, task, context="VP001/" + run)

    assert len(blocks) == task.n_blocks_per_run
    for b in blocks:
        assert len(b.trials) == len(b.trial_labels)


def test_nback_structure_matches_documented_design():
    """9 blocks, 3 per level, 20 trials each, target/non-target only for 2- and 3-back."""
    markers = shin2018.parse_vmrk(SUBJECT_DIR / "eeg" / "nback1.vmrk")
    blocks = shin2018.segment_blocks(markers, shin2018.NBACK, context="VP001/nback1")

    by_condition: dict[str, list] = {}
    for b in blocks:
        by_condition.setdefault(b.condition, []).append(b)

    assert sorted(by_condition) == ["0-back", "2-back", "3-back"]
    for condition, blks in by_condition.items():
        assert len(blks) == 3, condition
        for b in blks:
            assert len(b.trials) == 20, (condition, len(b.trials))

    # 0-back carries a single stimulus code; the published table's "S32" never appears.
    zero_labels = {lab for b in by_condition["0-back"] for lab in b.trial_labels}
    assert zero_labels == {"stimulus"}
    for condition in ("2-back", "3-back"):
        labels = {lab for b in by_condition[condition] for lab in b.trial_labels}
        assert labels == {"target", "nontarget"}


def test_eeg_raw_loads_with_expected_montage():
    raw = shin2018.load_eeg_run(SUBJECT_DIR, "nback1", preload=False)
    assert raw.info["sfreq"] == shin2018.EEG_SFREQ
    assert len(raw.ch_names) == 30
    eog = [ch for ch, t in zip(raw.ch_names, raw.get_channel_types()) if t == "eog"]
    assert set(eog) == set(shin2018.EOG_CHANNELS)


def test_word_runs_have_one_bookkeeping_trigger():
    """S208 is undocumented; assert it appears exactly once, so dropping it is safe."""
    for run in shin2018.WORD.runs:
        codes = [m.code for m in shin2018.parse_vmrk(SUBJECT_DIR / "eeg" / (run + ".vmrk"))]
        assert codes.count(208) == 1, (run, codes.count(208))
        assert codes.count(16) == 10 and codes.count(32) == 10, run


@pytest.mark.parametrize("run", ["nback1", "gonogo1", "word1"])
def test_nirx_reads_expected_geometry(run):
    rec = shin2018.read_nirs_run(SUBJECT_DIR, run)
    assert rec.n_channels == 36, rec.n_channels
    assert rec.wavelengths == (760.0, 850.0)
    assert rec.n_sources == 16 and rec.n_detectors == 16
    assert abs(rec.sfreq - shin2018.NIRS_SFREQ) < 0.01
    assert rec.data.shape == (2, 36, rec.n_samples)
    assert (rec.distances_mm == 30.0).all()
    assert rec.run_name == run
    assert np.isfinite(rec.data).all()
    assert (rec.data > 0).all(), "raw CW amplitude must be positive for the log in OD"


def test_nirx_to_mne_roundtrip():
    raw = shin2018.load_nirs_run(SUBJECT_DIR, "nback1")
    assert len(raw.ch_names) == 72
    assert abs(raw.info["sfreq"] - shin2018.NIRS_SFREQ) < 0.01

    import mne

    dists = mne.preprocessing.nirs.source_detector_distances(raw.info)
    assert np.allclose(dists, 0.030, atol=1e-6), dists[:5]

    od = mne.preprocessing.nirs.optical_density(raw)
    hb = mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0)
    types = set(hb.get_channel_types())
    assert types == {"hbo", "hbr"}, types
    assert np.isfinite(hb.get_data()).all()


@pytest.mark.parametrize("run", shin2018.NBACK.runs)
def test_nirs_codes_independently_confirm_eeg_block_mapping(run):
    """112/128/144 -> 0/2/3-back was reverse-engineered; NIRS codes 7/8/9 confirm it."""
    check = shin2018.cross_check_nirs_blocks(SUBJECT_DIR, run)
    assert check.agree, check.mismatches
    conditions = list(check.from_eeg)
    assert len(conditions) == 9
    assert sorted(set(conditions)) == ["0-back", "2-back", "3-back"]
    for c in set(conditions):
        assert conditions.count(c) == 3


def test_session_map_resolves_bad_inf_notes():
    """VP016 mislabels a word run as nback2, VP024 leaves one note empty.

    Both are recovered from the trigger sequence rather than dropped or guessed by
    timestamp. Skipped unless those subjects are present.
    """
    for vp, run, expected_dir in (("VP016", "nback2", "2016-06-08_002"),
                                  ("VP016", "word2", "2016-06-08_005"),
                                  ("VP024", "nback2", "2016-06-16_011")):
        subject_dir = PATHS.interim / "shin2018" / vp
        if not (subject_dir / "nirs").is_dir():
            pytest.skip(vp + " not extracted")
        assert shin2018.nirs_session_map(subject_dir)[run].name == expected_dir


@pytest.mark.parametrize("run", shin2018.ALL_RUNS)
def test_devices_are_tightly_synchronised(run):
    """The claim that a multi-second EEG-fNIRS lag is physiological rests on this."""
    align = shin2018.align_run(SUBJECT_DIR, run)
    assert align.n_events >= 6
    # One parallel-port trigger to both amplifiers: residuals must be far below the
    # ~5-7 s haemodynamic lag we intend to model.
    assert align.max_abs_residual < 0.1, (run, align)
