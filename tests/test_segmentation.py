import json
import sys

import nibabel as nib
import numpy as np
import pytest

from fissure_annotation.imaging.lobes import LEFT_LOWER, LEFT_UPPER
from fissure_annotation.segmentation import (
    LABEL_IDS,
    LUNG_LOBES,
    TotalSegmentatorSettings,
    run_totalsegmentator,
    totalsegmentator_arguments,
)

FAKE_TOTALSEGMENTATOR = """
import json, os, sys
import nibabel as nib
import numpy as np

arguments = sys.argv[1:]
out_path = arguments[arguments.index("-o") + 1]
with open(out_path + ".call.json", "w") as file:
    json.dump({"arguments": arguments, "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}, file)
if os.environ.get("FAKE_FAIL"):
    sys.stderr.write("CUDA out of memory")
    sys.exit(3)
volume = np.zeros((4, 4, 4), dtype=np.uint8)
for position, label in enumerate(json.loads(os.environ["FAKE_IDS"])):
    volume[position] = label
if not os.environ.get("FAKE_NO_FILE"):
    nib.save(nib.Nifti1Image(volume, np.eye(4)), out_path)
"""


@pytest.fixture
def fake_settings(tmp_path):
    script = tmp_path / "fake_totalsegmentator.py"
    script.write_text(FAKE_TOTALSEGMENTATOR)
    return TotalSegmentatorSettings(command=[sys.executable, str(script)])


def last_call(out_path):
    return json.loads((out_path.parent / (out_path.name + ".call.json")).read_text())


def test_the_command_asks_for_one_multilabel_file_with_only_the_wanted_structures(
    tmp_path,
):
    arguments = totalsegmentator_arguments(
        tmp_path / "scan.nii.gz",
        ["lung_upper_lobe_left", "lung_lower_lobe_left"],
        tmp_path / "out.nii.gz",
        TotalSegmentatorSettings(command=["TotalSegmentator"]),
    )
    assert arguments[0] == "TotalSegmentator"
    assert arguments[arguments.index("-i") + 1] == str(tmp_path / "scan.nii.gz")
    assert arguments[arguments.index("-o") + 1] == str(tmp_path / "out.nii.gz")
    assert "--ml" in arguments and arguments[arguments.index("--task") + 1] == "total"
    subset = arguments.index("--roi_subset")
    assert arguments[subset + 1 : subset + 3] == [
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
    ]
    assert arguments[arguments.index("-d") + 1] == "gpu"


def test_the_lobe_ids_are_the_ones_the_imaging_layer_uses():
    assert LABEL_IDS["lung_upper_lobe_left"] == LEFT_UPPER
    assert LABEL_IDS["lung_lower_lobe_left"] == LEFT_LOWER
    assert sorted(LABEL_IDS) == sorted(LUNG_LOBES)


def test_the_segmenter_runs_the_command_and_returns_the_file(
    tmp_path, fake_settings, monkeypatch
):
    monkeypatch.setenv("FAKE_IDS", json.dumps([LEFT_UPPER, LEFT_LOWER]))
    out_path = tmp_path / "nested" / "lobes.nii.gz"
    result = run_totalsegmentator(
        tmp_path / "scan.nii.gz", LUNG_LOBES, out_path, fake_settings
    )
    assert result == out_path and out_path.exists()
    assert last_call(out_path)["arguments"][-1] == "gpu"


def test_a_gpu_index_sets_the_visible_device_for_the_call_only(
    tmp_path, fake_settings, monkeypatch
):
    monkeypatch.setenv("FAKE_IDS", json.dumps([LEFT_UPPER]))
    out_path = tmp_path / "lobes.nii.gz"
    settings = fake_settings.model_copy(update={"gpu_index": 1})
    run_totalsegmentator(tmp_path / "scan.nii.gz", LUNG_LOBES, out_path, settings)
    assert last_call(out_path)["cuda"] == "1"


def test_labels_that_were_not_asked_for_are_refused(
    tmp_path, fake_settings, monkeypatch
):
    monkeypatch.setenv("FAKE_IDS", json.dumps([LEFT_UPPER, 99]))
    with pytest.raises(ValueError, match="99"):
        run_totalsegmentator(
            tmp_path / "scan.nii.gz",
            LUNG_LOBES,
            tmp_path / "lobes.nii.gz",
            fake_settings,
        )


def test_a_structure_that_was_not_found_is_not_an_error(
    tmp_path, fake_settings, monkeypatch
):
    monkeypatch.setenv(
        "FAKE_IDS", json.dumps([LEFT_UPPER])
    )  # the other lobes are missing
    out_path = run_totalsegmentator(
        tmp_path / "scan.nii.gz", LUNG_LOBES, tmp_path / "lobes.nii.gz", fake_settings
    )
    assert set(np.unique(np.asarray(nib.load(out_path).dataobj))) == {0, LEFT_UPPER}


def test_a_failing_command_raises_with_the_end_of_its_output(
    tmp_path, fake_settings, monkeypatch
):
    monkeypatch.setenv("FAKE_FAIL", "1")
    with pytest.raises(RuntimeError, match="(?s)exit code 3.*CUDA out of memory"):
        run_totalsegmentator(
            tmp_path / "scan.nii.gz",
            LUNG_LOBES,
            tmp_path / "lobes.nii.gz",
            fake_settings,
        )


def test_a_command_that_writes_nothing_raises(tmp_path, fake_settings, monkeypatch):
    monkeypatch.setenv("FAKE_IDS", json.dumps([LEFT_UPPER]))
    monkeypatch.setenv("FAKE_NO_FILE", "1")
    with pytest.raises(FileNotFoundError):
        run_totalsegmentator(
            tmp_path / "scan.nii.gz",
            LUNG_LOBES,
            tmp_path / "lobes.nii.gz",
            fake_settings,
        )
