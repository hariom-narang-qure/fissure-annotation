import json

import pytest

from fissure_annotation import cases
from fissure_annotation.cases import CasesSettings, case_files, picked_slices, read_picks


@pytest.fixture
def two_cases(tmp_path, monkeypatch):
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps({"scan1": ["a.nii.gz", "lobes1"], "scan2": ["b.nii.gz", "lobes2"]}))
    for name in ("scan1", "scan2"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "picks.json").write_text(json.dumps([["coronal", 10], ["axial", 20], ["axial", 30]]))
    monkeypatch.setattr(cases, "OUT", tmp_path)
    monkeypatch.setattr(cases, "load_case", lambda scan, lobes: f"case:{scan}")
    return cases_file


def test_every_picked_slice_of_every_case_is_a_job(two_cases):
    jobs = picked_slices(CasesSettings(cases=two_cases))
    assert [(j[1], j[2], j[3]) for j in jobs][:3] == [("scan1", "coronal", 10), ("scan1", "axial", 20), ("scan1", "axial", 30)]
    assert len(jobs) == 6


def test_only_filters_cases_by_name(two_cases):
    assert set(case_files(CasesSettings(cases=two_cases, only="scan2"))) == {"scan2"}
    assert {j[1] for j in picked_slices(CasesSettings(cases=two_cases, only="scan2"))} == {"scan2"}


def test_max_jobs_gives_a_quick_smoke_run(two_cases):
    assert len(picked_slices(CasesSettings(cases=two_cases, max_jobs=2))) == 2


def test_picks_are_plane_and_index_pairs(two_cases):
    assert read_picks("scan1")[0] == ("coronal", 10)
