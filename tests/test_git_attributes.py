# Copyright 2018 Davide Spadini
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
Tests for historical .gitattributes support (Git.get_attributes and the
attribute-based analysis views on Commit).

The repository used by these tests is built on the fly so that its
.gitattributes files *change over time*: classifying historical changes
must follow the rules in effect at each commit, not the current ones.
"""
import os
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pytest
from git import Repo

from pydriller.domain.commit import Commit
from pydriller.git import Git

ROOT_ATTRIBUTES_C1 = """\
# root rules
*.py lang=python
gen.py linguist-generated
vendor/*.js linguist-vendored
/anchored.txt rootonly=yes
docs/** draft
.hidden dot=keep
"""

PKG_ATTRIBUTES = """\
*.py lang=pkgpython
special linguist-generated
"""

ROOT_ATTRIBUTES_C2 = """\
# rules changed later in history
gen.py linguist-generated
*.py linguist-generated
vendor/lib.js -linguist-vendored
/anchored.txt rootonly=yes
docs/** draft
"""

C1_FILES: Dict[str, str] = {
    ".gitattributes": ROOT_ATTRIBUTES_C1,
    "pkg/.gitattributes": PKG_ATTRIBUTES,
    "app.py": "print('app')\nline2\n",
    "gen.py": "GENERATED CONTENT\nline2\n",
    "readme.md": "# readme\nline2\n",
    "vendor/lib.js": "var x = 1;\nline2\n",
    "anchored.txt": "root\nline2\n",
    "sub/anchored.txt": "nested\nline2\n",
    "docs/readme.md": "docs\nline2\n",
    "docs/chapter/intro.md": "intro\nline2\n",
    "pkg/a.py": "print('a')\nline2\n",
    "pkg/special": "special\nline2\n",
}

C2_FILES: Dict[str, str] = {
    ".gitattributes": ROOT_ATTRIBUTES_C2,
    "app.py": "print('app')\nline2\nline3\n",
    "gen.py": "GENERATED CONTENT\nline2\nline3\n",
    "vendor/lib.js": "var x = 1;\nline2\nline3\n",
    "readme.md": "# readme\nline2\nline3\n",
}


def _write_files(folder: Path, files: Dict[str, str]) -> None:
    for name, content in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def _commit(repo: Repo, folder: Path, files: Dict[str, str],
            message: str) -> str:
    _write_files(folder, files)
    repo.index.add([str(folder / name) for name in files])
    return repo.index.commit(message).hexsha


@pytest.fixture
def attr_repo(tmp_path: Path) -> Iterator[Tuple[Git, List[Commit], Repo]]:
    repo = Repo.init(tmp_path)
    writer = repo.config_writer()
    writer.set_value("user", "name", "Test").release()
    writer = repo.config_writer()
    writer.set_value("user", "email", "test@test.com").release()

    c1 = _commit(repo, tmp_path, C1_FILES, "initial rules")

    c2 = _commit(repo, tmp_path, C2_FILES, "rules evolve")

    # C3 renames the vendored file and deletes a generated one
    os.rename(tmp_path / "vendor" / "lib.js", tmp_path / "vendor" / "lib2.js")
    repo.index.remove(["vendor/lib.js"])
    repo.index.add([str(tmp_path / "vendor" / "lib2.js")])
    os.remove(tmp_path / "pkg" / "special")
    repo.index.remove(["pkg/special"])
    c3 = repo.index.commit("rename and delete").hexsha

    gr = Git(str(tmp_path))
    commits = list(gr.get_list_commits())
    assert [c.hash for c in commits] == [c1, c2, c3]
    yield gr, commits, repo
    gr.clear()


def _names(modifications) -> set:
    return {mod.filename for mod in modifications}


# ---------------------------------------------------------------------------
# Git.get_attributes: historical API
# ---------------------------------------------------------------------------

def test_attributes_follow_the_history(attr_repo):
    gr, (c1, c2, _c3), _ = attr_repo

    # app.py: at c1 it just carries a custom value, at c2 it is generated
    assert gr.get_attributes(c1.hash, "app.py") == {"lang": "python"}
    assert gr.get_attributes(c2.hash, "app.py") == {"linguist-generated": True}

    # the vendor rule disappears (unset) between c1 and c2
    assert gr.get_attributes(c1.hash, "vendor/lib.js") == {
        "linguist-vendored": True}
    assert gr.get_attributes(c2.hash, "vendor/lib.js") == {
        "linguist-vendored": False}


def test_attributes_ignore_the_current_working_tree(attr_repo):
    gr, (c1, c2, _c3), repo = attr_repo

    # Pretend "today" the project classifies everything differently, but
    # leave the new rules uncommitted in the working tree.
    (Path(repo.working_tree_dir) / ".gitattributes").write_text(
        "*.js linguist-vendored\n*.md linguist-generated\n"
        "*.py -linguist-generated\n")

    # Historical answers must be identical to the committed rules.
    assert gr.get_attributes(c1.hash, "app.py") == {"lang": "python"}
    assert gr.get_attributes(c2.hash, "app.py") == {"linguist-generated": True}
    assert gr.get_attributes(c1.hash, "vendor/lib.js") == {
        "linguist-vendored": True}
    assert gr.get_attributes(c2.hash, "vendor/lib.js") == {
        "linguist-vendored": False}
    assert gr.get_attributes(c1.hash, "readme.md") == {}


def test_attributes_states(attr_repo):
    gr, (c1, _c2, _c3), _ = attr_repo

    assert gr.get_attributes(c1.hash, "gen.py")["linguist-generated"] is True
    assert gr.get_attributes(c1.hash, "app.py")["lang"] == "python"
    # a path with no rule at all is an empty dict
    assert gr.get_attributes(c1.hash, "no-rule.txt") == {}


def test_nested_gitattributes_override_root(attr_repo):
    gr, (c1, _c2, _c3), _ = attr_repo

    # the nested *.py rule overrides the value set at the root
    assert gr.get_attributes(c1.hash, "pkg/a.py") == {"lang": "pkgpython"}
    # a basename rule in pkg/.gitattributes only applies below pkg/
    assert gr.get_attributes(c1.hash, "pkg/special") == {
        "linguist-generated": True}
    assert "linguist-generated" not in gr.get_attributes(c1.hash, "special")


def test_patterns_anchoring_and_recursion(attr_repo):
    gr, (c1, _c2, _c3), _ = attr_repo

    assert gr.get_attributes(c1.hash, "anchored.txt") == {"rootonly": "yes"}
    assert gr.get_attributes(c1.hash, "sub/anchored.txt") == {}

    assert gr.get_attributes(c1.hash, "docs/readme.md") == {"draft": True}
    assert gr.get_attributes(c1.hash, "docs/chapter/intro.md") == {
        "draft": True}
    assert gr.get_attributes(c1.hash, "readme.md") == {}


def test_path_normalization(attr_repo):
    gr, (c1, _c2, _c3), _ = attr_repo

    assert gr.get_attributes(c1.hash, ".hidden") == {"dot": "keep"}
    assert gr.get_attributes(c1.hash, "./.hidden") == {"dot": "keep"}
    assert gr.get_attributes(c1.hash, "./app.py") == {"lang": "python"}


def test_attributes_accept_gitpython_commit(attr_repo):
    gr, (c1, _c2, _c3), repo = attr_repo

    assert gr.get_attributes(repo.commit(c1.hash), "app.py") == {
        "lang": "python"}


# ---------------------------------------------------------------------------
# Commit: independent attribute-based analysis views
# ---------------------------------------------------------------------------

def test_views_on_initial_commit(attr_repo):
    _gr, (c1, _c2, _c3), _ = attr_repo

    assert len(c1.modified_files) == 12
    assert _names(c1.vendored_files) == {"lib.js"}
    assert _names(c1.generated_files) == {"gen.py", "special"}


def test_views_follow_historical_rules(attr_repo):
    _gr, (_c1, c2, _c3), _ = attr_repo

    assert _names(c2.generated_files) == {"app.py", "gen.py"}
    assert c2.vendored_files == []

    # unset attribute
    assert _names(c2.modified_files_matching_attributes(
        {"linguist-vendored": False})) == {"lib.js"}

    # unspecified attribute: everything except lib.js
    assert _names(c2.modified_files_matching_attributes(
        {"linguist-vendored": None})) == {
        ".gitattributes", "app.py", "gen.py", "readme.md"}

    # value matching + a second simultaneous condition
    assert _names(c2.modified_files_matching_attributes(
        {"linguist-generated": True, "lang": None})) == {"app.py", "gen.py"}


def test_views_keep_the_full_changeset_and_statistics(attr_repo):
    _gr, (_c1, c2, _c3), _ = attr_repo

    full = c2.modified_files
    assert len(full) == 5
    assert len(c2.generated_files) == 2
    assert len(c2.vendored_files) == 0

    # The view is a different list and does not mutate modified_files.
    assert c2.generated_files is not c2.modified_files
    assert len(c2.modified_files) == 5

    # Statistics keep representing the whole changeset.
    assert c2.files == len(full)
    assert c2.insertions == sum(mod.added_lines for mod in full)
    assert c2.deletions == sum(mod.deleted_lines for mod in full)
    assert c2.lines == c2.insertions + c2.deletions


def test_views_on_rename_and_delete(attr_repo):
    _gr, (_c1, _c2, c3), _ = attr_repo

    # The renamed file is evaluated with its new path against c3 rules,
    # the deleted one with its old path; neither call must raise.
    assert len(c3.modified_files) == 2
    assert c3.vendored_files == []
    assert _names(c3.generated_files) == {"special"}


def test_views_ignore_uncommitted_working_tree_rules(attr_repo):
    gr, (c1, c2, _c3), repo = attr_repo

    (Path(repo.working_tree_dir) / ".gitattributes").write_text(
        "*.js linguist-vendored\n*.md linguist-generated\n")

    assert _names(gr.get_commit(c2.hash).vendored_files) == set()
    assert _names(gr.get_commit(c1.hash).vendored_files) == {"lib.js"}


def test_views_through_repository_traversal(attr_repo):
    from pydriller.repository import Repository

    _gr, (_c1, c2, _c3), repo = attr_repo

    found = 0
    for commit in Repository(repo.working_tree_dir, single=c2.hash) \
            .traverse_commits():
        found += 1
        assert _names(commit.generated_files) == {"app.py", "gen.py"}
        assert commit.vendored_files == []
        assert len(commit.modified_files) == 5
        assert commit.files == 5
    assert found == 1


def test_view_after_traversal_ended(attr_repo):
    from pydriller.repository import Repository

    _gr, (_c1, c2, _c3), repo = attr_repo

    commits = list(Repository(repo.working_tree_dir, single=c2.hash)
                   .traverse_commits())
    commit = commits[0]

    # The Git wrapper is disposed with the traversal, but the view can
    # still read historical attributes from the commit objects.
    assert _names(commit.generated_files) == {"app.py", "gen.py"}
    assert _names(commit.vendored_files) == set()
    assert len(commit.modified_files) == 5
