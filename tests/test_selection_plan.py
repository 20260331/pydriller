# Copyright 2024 Davide Spadini
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for the fixed selection plan and the verifiable resume cursor.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from pydriller import Git, Repository, ResumeCursor, SelectionPlan, SelectionPlanError

SMALL_REPO = "test-repos/small_repo"
TAGS_REPO = "test-repos/tags"
MERGED_REPO = "test-repos/branches_merged"


def _hashes(path_to_repo: str, **kwargs):
    return [
        commit.hash
        for commit in Repository(path_to_repo=path_to_repo, **kwargs).traverse_commits()
    ]


def _planned_repository(tmp_path: Path, name: str = "plan.json", **kwargs):
    plan_path = tmp_path / name
    cursor_path = tmp_path / "cursor.json"
    repository = Repository(
        path_to_repo=SMALL_REPO,
        selection_plan=str(plan_path),
        resume_cursor=str(cursor_path),
        **kwargs
    )
    return repository, plan_path, cursor_path


# ---------------------------------------------------------------------------
# SelectionPlan (unit tests, no repository)
# ---------------------------------------------------------------------------

class TestSelectionPlan:
    def test_plan_keeps_order_and_identity(self):
        first = SelectionPlan(["a", "b", "c"])
        second = SelectionPlan(["a", "b", "c"])
        reordered = SelectionPlan(["c", "b", "a"])

        assert first.plan_id == second.plan_id
        assert first.plan_id != reordered.plan_id
        assert list(first) == ["a", "b", "c"]
        assert len(first) == 3
        assert first[0] == "a"
        assert first.index_of("b") == 1

    def test_plan_id_covers_frozen_filters(self):
        without_tags = SelectionPlan(["a"], tagged_commits=None)
        with_tags = SelectionPlan(["a"], tagged_commits={"t1"})
        with_other_tags = SelectionPlan(["a"], tagged_commits={"t2"})

        assert without_tags.plan_id != with_tags.plan_id
        assert with_tags.plan_id != with_other_tags.plan_id

    def test_duplicate_commits_are_rejected(self):
        with pytest.raises(SelectionPlanError):
            SelectionPlan(["a", "b", "a"])

    def test_missing_commits_reports_unknown_hashes(self):
        plan = SelectionPlan(["a", "b", "c"])
        assert plan.missing_commits({"a", "c"}) == ["b"]
        assert plan.missing_commits({"a", "b", "c"}) == []

    def test_plan_save_and_load_roundtrip(self, tmp_path):
        plan = SelectionPlan(
            ["a", "b"], repo="/repo", head="a",
            tagged_commits={"t"}, filepath_commits={"b"},
        )
        plan_path = tmp_path / "plan.json"
        plan.save(plan_path)

        reloaded = SelectionPlan.load(plan_path)
        assert reloaded.commits == ("a", "b")
        assert reloaded.repo == "/repo"
        assert reloaded.head == "a"
        assert reloaded.tagged_commits == frozenset({"t"})
        assert reloaded.filepath_commits == frozenset({"b"})
        assert reloaded.plan_id == plan.plan_id

    def test_tampered_plan_is_rejected(self, tmp_path):
        plan = SelectionPlan(["a", "b"])
        plan_path = tmp_path / "plan.json"
        plan.save(plan_path)

        data = json.loads(plan_path.read_text())
        data["commits"] = ["a", "c"]
        plan_path.write_text(json.dumps(data))

        with pytest.raises(SelectionPlanError):
            SelectionPlan.load(plan_path)

    def test_unsupported_version_is_rejected(self, tmp_path):
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps({"version": 999, "plan_id": "x",
                                         "commits": []}))
        with pytest.raises(SelectionPlanError):
            SelectionPlan.load(plan_path)


# ---------------------------------------------------------------------------
# ResumeCursor (unit tests, no repository)
# ---------------------------------------------------------------------------

class TestResumeCursor:
    def test_in_order_delivery(self):
        plan = SelectionPlan(["a", "b"])
        cursor = ResumeCursor(plan)

        assert cursor.position == 0
        assert not cursor.is_complete
        assert cursor.mark_delivered("a") is True
        assert cursor.position == 1
        assert cursor.pending_commits() == ["b"]
        assert cursor.mark_delivered("b") is True
        assert cursor.position == 2
        assert cursor.is_complete

    def test_out_of_order_delivery_does_not_skip_the_gap(self):
        plan = SelectionPlan(["a", "b", "c", "d"])
        cursor = ResumeCursor(plan)

        # A worker delivers future commits first: parked, no advancement.
        assert cursor.mark_delivered("b") is False
        assert cursor.mark_delivered("d") is False
        assert cursor.position == 0
        assert cursor.delivered == {"b", "d"}
        # b and d are already acked (parked), so only the gaps must replay.
        assert cursor.pending_commits() == ["a", "c"]
        assert cursor.is_pending(0, "a") is True
        assert cursor.is_pending(1, "b") is False

        # Filling the first gap advances through the contiguous tail only.
        assert cursor.mark_delivered("a") is True
        assert cursor.position == 2
        assert cursor.delivered == {"d"}

        # c closes the remaining gap and swallows d.
        assert cursor.mark_delivered("c") is True
        assert cursor.position == 4
        assert cursor.delivered == frozenset()
        assert cursor.is_complete

    def test_unknown_commit_cannot_advance_the_cursor(self):
        cursor = ResumeCursor(SelectionPlan(["a", "b"]))
        with pytest.raises(SelectionPlanError):
            cursor.mark_delivered("zzz")
        assert cursor.position == 0

    def test_cursor_cannot_move_backwards(self):
        plan = SelectionPlan(["a", "b"])
        cursor = ResumeCursor(plan)
        cursor.mark_delivered("a")

        with pytest.raises(SelectionPlanError):
            cursor.mark_delivered("a")
        assert cursor.position == 1

    def test_complete_plan_rejects_further_acks(self):
        plan = SelectionPlan(["a"])
        cursor = ResumeCursor(plan)
        cursor.mark_delivered("a")
        with pytest.raises(SelectionPlanError):
            cursor.mark_delivered("a")

    def test_invalid_initial_state_is_rejected(self):
        plan = SelectionPlan(["a", "b"])

        with pytest.raises(SelectionPlanError):
            ResumeCursor(plan, position=5)
        with pytest.raises(SelectionPlanError):
            ResumeCursor(plan, position=-1)
        # 'a' is already part of the contiguous prefix
        with pytest.raises(SelectionPlanError):
            ResumeCursor(plan, position=1, delivered=["a"])
        with pytest.raises(SelectionPlanError):
            ResumeCursor(plan, delivered=["zzz"])

    def test_cursor_bound_to_another_plan_is_rejected(self):
        plan_a = SelectionPlan(["a", "b"])
        plan_b = SelectionPlan(["c", "d"])
        cursor = ResumeCursor(plan_a)

        cursor.mark_delivered("a")
        with pytest.raises(SelectionPlanError):
            cursor.bind(plan_b)
        # Binding to the same plan is fine and keeps the state.
        cursor.bind(plan_a)
        assert cursor.position == 1

    def test_save_and_load_roundtrip(self, tmp_path):
        plan = SelectionPlan(["a", "b", "c"])
        cursor = ResumeCursor(plan)
        cursor.mark_delivered("a")
        cursor.mark_delivered("c")  # out of order, parked

        cursor_path = tmp_path / "cursor.json"
        cursor.save(cursor_path)

        reloaded = ResumeCursor.load(cursor_path, plan)
        assert reloaded.position == 1
        assert reloaded.delivered == {"c"}
        assert not reloaded.is_complete

    def test_loaded_cursor_auto_persists_advancements(self, tmp_path):
        plan = SelectionPlan(["a", "b"])
        cursor_path = tmp_path / "cursor.json"
        ResumeCursor(plan).save(cursor_path)

        cursor = ResumeCursor.load(cursor_path, plan)
        cursor.mark_delivered("a")

        on_disk = json.loads(cursor_path.read_text())
        assert on_disk["position"] == 1

    def test_tampered_cursor_is_rejected(self, tmp_path):
        plan = SelectionPlan(["a", "b"])
        cursor_path = tmp_path / "cursor.json"
        ResumeCursor(plan).save(cursor_path)

        data = json.loads(cursor_path.read_text())
        data["position"] = 1
        cursor_path.write_text(json.dumps(data))

        with pytest.raises(SelectionPlanError):
            ResumeCursor.load(cursor_path, plan)

    def test_cursor_from_another_plan_fails_to_load(self, tmp_path):
        plan_a = SelectionPlan(["a", "b"])
        plan_b = SelectionPlan(["a", "b", "c"])
        cursor_path = tmp_path / "cursor.json"
        ResumeCursor(plan_a).save(cursor_path)

        with pytest.raises(SelectionPlanError):
            ResumeCursor.load(cursor_path, plan_b)


# ---------------------------------------------------------------------------
# Repository integration
# ---------------------------------------------------------------------------

class TestPlannedTraversal:
    def test_plan_freezes_the_ordered_selection(self, tmp_path):
        baseline = _hashes(SMALL_REPO)

        repository, plan_path, cursor_path = _planned_repository(tmp_path)
        delivered = [commit.hash for commit in repository.traverse_commits()]

        assert delivered == baseline

        plan_data = json.loads(plan_path.read_text())
        assert plan_data["commits"] == baseline
        assert plan_data["head"] == baseline[-1]
        assert plan_data["repo"].endswith("small_repo")

        cursor = ResumeCursor.load(cursor_path, SelectionPlan.load(plan_path))
        assert cursor.position == len(baseline)
        assert cursor.is_complete

    def test_default_cursor_file_next_to_plan(self, tmp_path):
        plan_path = tmp_path / "frozen.json"
        repository = Repository(
            path_to_repo=SMALL_REPO, selection_plan=str(plan_path)
        )
        list(repository.traverse_commits())

        assert plan_path.exists()
        assert (tmp_path / "frozen.cursor.json").exists()

    def test_resume_replays_remaining_planned_commits(self, tmp_path):
        baseline = _hashes(SMALL_REPO)
        repository, plan_path, cursor_path = _planned_repository(tmp_path)

        delivered = []
        for commit in repository.traverse_commits():
            delivered.append(commit.hash)
            if len(delivered) == 3:
                break  # simulate a crash while processing commit #3

        # The commit in flight when the crash happened is not acknowledged,
        # so the cursor stops at the previous commit.
        crashed_cursor = json.loads(cursor_path.read_text())
        assert crashed_cursor["position"] == 2

        resumed = _hashes(
            SMALL_REPO,
            selection_plan=str(plan_path),
            resume_cursor=str(cursor_path),
        )
        # At-least-once: the in-flight commit is delivered again, then the
        # untouched suffix follows; nothing gets skipped or duplicated twice.
        assert delivered[:-1] + resumed == baseline
        assert resumed[0] == baseline[2]

        # Resuming a finished traversal delivers nothing.
        again = _hashes(
            SMALL_REPO,
            selection_plan=str(plan_path),
            resume_cursor=str(cursor_path),
        )
        assert again == []

    def test_resume_from_cursor_position(self, tmp_path):
        baseline = _hashes(SMALL_REPO)
        repository, plan_path, cursor_path = _planned_repository(tmp_path)
        list(repository.traverse_commits())

        # Pretend a previous run stopped after 2 commits.
        plan = SelectionPlan.load(plan_path)
        ResumeCursor(plan, position=2).save(cursor_path)

        resumed = _hashes(
            SMALL_REPO,
            selection_plan=str(plan_path),
            resume_cursor=str(cursor_path),
        )
        assert resumed == baseline[2:]

    def test_resume_with_out_of_order_cursor(self, tmp_path):
        baseline = _hashes(SMALL_REPO)
        repository, plan_path, cursor_path = _planned_repository(tmp_path)
        list(repository.traverse_commits())

        # Worker acked commit #3 out of order while commits #1-2 were pending.
        plan = SelectionPlan.load(plan_path)
        cursor = ResumeCursor(plan)
        cursor.mark_delivered(baseline[2])
        cursor.save(cursor_path)

        resumed = _hashes(
            SMALL_REPO,
            selection_plan=str(plan_path),
            resume_cursor=str(cursor_path),
        )
        assert resumed == baseline[:2] + baseline[3:]

    def test_branch_moving_does_not_change_the_plan(self, tmp_path):
        baseline = _hashes(SMALL_REPO)
        repo_copy = tmp_path / "small_repo"
        shutil.copytree(SMALL_REPO, str(repo_copy))

        plan_path = tmp_path / "plan.json"
        cursor_path = tmp_path / "cursor.json"
        first = _hashes(
            str(repo_copy),
            selection_plan=str(plan_path),
            resume_cursor=str(cursor_path),
        )
        assert first == baseline

        # A new run starts from scratch, but meanwhile the branch moved.
        cursor_path.unlink()
        (repo_copy / "newfile.txt").write_text("change\n")
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "tester",
            "GIT_AUTHOR_EMAIL": "tester@test.org",
            "GIT_COMMITTER_NAME": "tester",
            "GIT_COMMITTER_EMAIL": "tester@test.org",
        }
        subprocess.run(["git", "-C", str(repo_copy), "add", "-A"],
                       check=True, env=env, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "-C", str(repo_copy), "commit", "-m", "new"],
                       check=True, env=env, stdout=subprocess.DEVNULL)

        # The live history now has 6 commits; the plan still says 5.
        assert len(_hashes(str(repo_copy))) == len(baseline) + 1
        resumed = _hashes(
            str(repo_copy),
            selection_plan=str(plan_path),
            resume_cursor=str(cursor_path),
        )
        assert resumed == baseline

    def test_missing_planned_commit_is_a_hard_error(self, tmp_path):
        baseline = _hashes(SMALL_REPO)
        plan = SelectionPlan(baseline[:2] + ["0" * 40])
        plan_path = tmp_path / "plan.json"
        plan.save(plan_path)

        with pytest.raises(SelectionPlanError):
            list(Repository(
                path_to_repo=SMALL_REPO,
                selection_plan=str(plan_path),
            ).traverse_commits())

    def test_cursor_of_another_plan_is_rejected(self, tmp_path):
        # Build a plan + cursor on the small repo.
        _, small_plan_path, small_cursor_path = _planned_repository(
            tmp_path, name="small.json"
        )
        list(Repository(
            path_to_repo=SMALL_REPO,
            selection_plan=str(small_plan_path),
            resume_cursor=str(small_cursor_path),
        ).traverse_commits())

        # A different selection (different repository, different plan id).
        other_plan_path = tmp_path / "other.json"
        other_cursor_path = tmp_path / "other.cursor.json"
        list(Repository(
            path_to_repo=MERGED_REPO,
            selection_plan=str(other_plan_path),
            resume_cursor=str(other_cursor_path),
        ).traverse_commits())

        with pytest.raises(SelectionPlanError):
            list(Repository(
                path_to_repo=SMALL_REPO,
                selection_plan=str(small_plan_path),
                resume_cursor=str(other_cursor_path),
            ).traverse_commits())

    def test_selection_plan_does_not_support_multiple_repos(self, tmp_path):
        with pytest.raises(SelectionPlanError):
            list(Repository(
                path_to_repo=[SMALL_REPO, MERGED_REPO],
                selection_plan=str(tmp_path / "plan.json"),
            ).traverse_commits())

    def test_cursor_without_plan_is_rejected(self, tmp_path):
        with pytest.raises(Exception):
            list(Repository(
                path_to_repo=SMALL_REPO,
                resume_cursor=str(tmp_path / "cursor.json"),
            ).traverse_commits())

    def test_filtered_commits_still_complete_the_cursor(self, tmp_path):
        # only_releases is applied in Python: planned commits that are not
        # tagged get filtered out but must advance the cursor anyway.
        baseline = _hashes(TAGS_REPO, only_releases=True)
        assert len(baseline) == 3

        plan_path = tmp_path / "plan.json"
        cursor_path = tmp_path / "cursor.json"
        delivered = _hashes(
            TAGS_REPO,
            only_releases=True,
            selection_plan=str(plan_path),
            resume_cursor=str(cursor_path),
        )
        assert delivered == baseline

        plan = SelectionPlan.load(plan_path)
        cursor = ResumeCursor.load(cursor_path, plan)
        assert len(plan) == 5
        assert cursor.is_complete
        assert plan.tagged_commits == frozenset(baseline)

    def test_filepath_filter_freezes_selection_and_resumes(self, tmp_path):
        baseline = _hashes(SMALL_REPO, filepath="file2.java")
        assert len(baseline) == 3

        plan_path = tmp_path / "plan.json"
        cursor_path = tmp_path / "cursor.json"
        repository = Repository(
            path_to_repo=SMALL_REPO,
            filepath="file2.java",
            selection_plan=str(plan_path),
            resume_cursor=str(cursor_path),
        )
        delivered = []
        for commit in repository.traverse_commits():
            delivered.append(commit.hash)
            if len(delivered) == 2:
                break

        resumed = _hashes(
            SMALL_REPO,
            filepath="file2.java",
            selection_plan=str(plan_path),
            resume_cursor=str(cursor_path),
        )
        assert delivered[:-1] + resumed == baseline

        plan = SelectionPlan.load(plan_path)
        assert plan.filepath_commits is not None
        assert set(baseline).issubset(plan.filepath_commits)

    def test_no_merge_filter_with_plan_and_resume(self, tmp_path):
        baseline = _hashes(MERGED_REPO, only_no_merge=True)

        plan_path = tmp_path / "plan.json"
        cursor_path = tmp_path / "cursor.json"
        delivered = _hashes(
            MERGED_REPO,
            only_no_merge=True,
            selection_plan=str(plan_path),
            resume_cursor=str(cursor_path),
        )
        assert delivered == baseline
        cursor = ResumeCursor.load(
            cursor_path, SelectionPlan.load(plan_path)
        )
        assert cursor.is_complete

    def test_in_memory_plan_and_cursor(self):
        git = Git(SMALL_REPO)
        hashes = [commit.hash for commit in git.get_list_commits()]
        head = git.repo.head.commit.hexsha
        git.clear()

        plan = SelectionPlan(hashes, repo=SMALL_REPO, head=head)
        cursor = ResumeCursor(plan)

        delivered = _hashes(
            SMALL_REPO, selection_plan=plan, resume_cursor=cursor
        )
        assert delivered == hashes
        assert cursor.is_complete
        assert cursor.position == len(hashes)

    def test_planned_traversal_with_workers(self, tmp_path):
        baseline = _hashes(SMALL_REPO)
        repository, plan_path, cursor_path = _planned_repository(
            tmp_path, num_workers=2
        )
        delivered = [commit.hash for commit in repository.traverse_commits()]
        assert delivered == baseline

        resumed = _hashes(
            SMALL_REPO,
            num_workers=2,
            selection_plan=str(plan_path),
            resume_cursor=str(cursor_path),
        )
        assert resumed == []

    def test_objects_exist_checks_the_object_store(self):
        git = Git(SMALL_REPO)
        hashes = [commit.hash for commit in git.get_list_commits()]
        try:
            found = git.objects_exist(hashes + ["0" * 40])
        finally:
            git.clear()
        assert found == set(hashes)
        assert "0" * 40 not in found
