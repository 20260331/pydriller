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
Tests for reading Git notes from configurable, pinned notes refs.
"""

import logging
import os
from pathlib import Path

import pytest
from git import Repo

from pydriller import Git, Repository
from pydriller.utils.conf import Conf


@pytest.fixture
def notes_repo(tmp_path):
    repo = Repo.init(tmp_path)
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "dev@test.com")
        cw.set_value("user", "name", "dev")

    tracked_file = tmp_path / "file.txt"
    tracked_file.write_text("a\n")
    repo.index.add(["file.txt"])
    c1 = repo.index.commit("commit 1")

    tracked_file.write_text("a\nb\n")
    repo.index.add(["file.txt"])
    c2 = repo.index.commit("commit 2")

    repo.git.notes("--ref=refs/notes/commits", "add", "-m", "classification v1", c1.hexsha)
    repo.git.notes("--ref=review", "add", "-m", "reviewed", c2.hexsha)

    return tmp_path, repo, c1, c2


def _by_hash(commits):
    return {commit.hash: commit for commit in commits}


def test_notes_disabled_by_default(notes_repo):
    path, _, _, _ = notes_repo
    commits = list(Repository(str(path)).traverse_commits())

    assert len(commits) == 2
    for commit in commits:
        assert commit.notes == {}
        assert commit.note is None


def test_read_note_from_default_ref(notes_repo):
    path, _, c1, c2 = notes_repo
    commits = _by_hash(Repository(str(path), notes_ref="refs/notes/commits").traverse_commits())

    assert commits[c1.hexsha].note == "classification v1\n"
    assert commits[c1.hexsha].notes == {"refs/notes/commits": "classification v1\n"}
    assert commits[c2.hexsha].notes == {}
    assert commits[c2.hexsha].note is None


def test_read_note_with_short_ref_name(notes_repo):
    path, _, c1, _ = notes_repo
    commits = _by_hash(Repository(str(path), notes_ref="commits").traverse_commits())

    # the short name is resolved to the full refs/notes/ name
    assert commits[c1.hexsha].notes == {"refs/notes/commits": "classification v1\n"}


def test_multiple_notes_refs(notes_repo):
    path, _, c1, c2 = notes_repo
    repo = Repository(str(path),
                      notes_ref=["refs/notes/commits", "review"])
    commits = _by_hash(repo.traverse_commits())

    assert commits[c1.hexsha].notes == {"refs/notes/commits": "classification v1\n"}
    assert commits[c2.hexsha].notes == {"refs/notes/review": "reviewed\n"}
    assert list(commits[c1.hexsha].notes.keys()) == ["refs/notes/commits"]


def test_duplicate_notes_refs(notes_repo):
    path, _, c1, _ = notes_repo
    # the same ref given with two spellings must be reported only once
    repo = Repository(str(path),
                      notes_ref=["commits", "refs/notes/commits"])
    commits = _by_hash(repo.traverse_commits())

    assert commits[c1.hexsha].notes == {"refs/notes/commits": "classification v1\n"}


def test_missing_notes_ref_does_not_crash(notes_repo, caplog):
    path, _, _, _ = notes_repo
    with caplog.at_level(logging.WARNING):
        commits = list(Repository(str(path), notes_ref="refs/notes/does-not-exist").traverse_commits())

    assert len(commits) == 2
    for commit in commits:
        assert commit.notes == {}
        assert commit.note is None
    assert "not found" in caplog.text


def test_notes_ref_is_pinned_when_traversal_starts(notes_repo):
    path, repo, c1, c2 = notes_repo

    traversal = iter(Repository(str(path), notes_ref="refs/notes/commits").traverse_commits())

    # commits are returned oldest-first; starting the traversal already
    # pinned the notes ref to its current object version
    first = next(traversal)
    assert first.hash == c1.hexsha

    # the notes ref moves while the traversal is in progress
    repo.git.notes("--ref=refs/notes/commits", "add", "-f",
                   "-m", "classification v2", c1.hexsha)

    assert first.note == "classification v1\n"
    second = next(traversal)
    assert second.hash == c2.hexsha
    assert second.notes == {}

    # a new traversal instead sees the updated notes
    fresh = _by_hash(Repository(str(path), notes_ref="refs/notes/commits").traverse_commits())
    assert fresh[c1.hexsha].note == "classification v2\n"


def test_notes_pinned_when_git_is_created(notes_repo):
    path, repo, c1, _ = notes_repo

    conf = Conf({
        "path_to_repo": str(path),
        "notes_refs": ["refs/notes/commits"],
    })
    git = Git(str(path), conf)

    commit = git.get_commit(c1.hexsha)
    assert commit.note == "classification v1\n"

    # moving the ref after the Git object was created does not affect it
    repo.git.notes("--ref=refs/notes/commits", "add", "-f",
                   "-m", "classification v2", c1.hexsha)
    assert commit.note == "classification v1\n"


def test_git_without_notes_configuration(notes_repo):
    path, _, c1, _ = notes_repo
    git = Git(str(path))
    commit = git.get_commit(c1.hexsha)
    assert commit.notes == {}
    assert commit.note is None


def test_fanout_notes_tree(notes_repo):
    path, repo, _, c2 = notes_repo

    # manually build a notes ref that stores the note in a 2/38 fan-out
    # subtree, the layout git uses when many notes accumulate
    note_file = path / "fanout-note.txt"
    note_file.write_text("fanout note\n")
    blob_sha = repo.git.hash_object("-w", str(note_file)).strip()
    index_file = str(Path(repo.git_dir) / "fanout-index")
    fanout_path = c2.hexsha[:2] + "/" + c2.hexsha[2:]
    try:
        with repo.git.custom_environment(GIT_INDEX_FILE=index_file):
            repo.git.update_index("--add", "--cacheinfo",
                                  f"100644,{blob_sha},{fanout_path}")
            tree_sha = repo.git.write_tree()
        notes_commit_sha = repo.git.commit_tree(tree_sha, m="fanout notes")
        repo.git.update_ref("refs/notes/fanout", notes_commit_sha)
    finally:
        if os.path.exists(index_file):
            os.remove(index_file)

    commits = _by_hash(Repository(str(path), notes_ref="fanout").traverse_commits())
    assert commits[c2.hexsha].note == "fanout note\n"
