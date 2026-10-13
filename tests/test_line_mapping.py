# Copyright 2026 Davide Spadini
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
Tests for the bidirectional line/range mapping (ModifiedFile.line_mapping)
and for the composition of adjacent mappings.
"""
from pathlib import Path

import pytest
from git import Repo

from pydriller import Git, ModifiedFile
from pydriller.domain.line_mapping import (
    AlignmentType,
    DiffBlock,
    LineMapping,
)

V1 = "L1\nL2\nL3\nL4\nL5\nL6\nL7\nL8\nL9\nL10\n"
V2 = (
    "L1\nIA1\nIA2\nL2\nL3\nL4-MOD\nL5\nL6\nL8\nL9\nL10\n"
)
V3 = (
    "TOP\nL1\nIA1\nIA2\nL2\nL3\nL4-MOD\nL5\nL6\nL8\nL9-CHANGED\nL10\n"
)

BLOCKS_V1_V2 = [
    DiffBlock(AlignmentType.EQUAL, 1, 1, 1, 1),
    DiffBlock(AlignmentType.INSERT, None, None, 2, 3),
    DiffBlock(AlignmentType.EQUAL, 2, 3, 4, 5),
    DiffBlock(AlignmentType.DELETE, 4, 4, None, None),
    DiffBlock(AlignmentType.INSERT, None, None, 6, 6),
    DiffBlock(AlignmentType.EQUAL, 5, 6, 7, 8),
    DiffBlock(AlignmentType.DELETE, 7, 7, None, None),
    DiffBlock(AlignmentType.EQUAL, 8, 10, 9, 11),
]

OLD_TO_NEW_V2 = {
    1: 1, 2: 4, 3: 5, 4: None, 5: 7,
    6: 8, 7: None, 8: 9, 9: 10, 10: 11,
}
NEW_TO_OLD_V2 = {
    1: 1, 2: None, 3: None, 4: 2, 5: 3, 6: None,
    7: 5, 8: 6, 9: 8, 10: 9, 11: 10,
}


def _commit_file(repo: Repo, filename: str, content: str,
                 message: str) -> str:
    path = Path(str(repo.working_tree_dir)) / filename
    path.write_text(content)
    repo.index.add([filename])
    return repo.index.commit(message).hexsha


@pytest.fixture()
def three_versions(tmp_path: Path):
    repo = Repo.init(str(tmp_path / "repo"))
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@test.com")
        cw.set_value("user", "name", "Test")
    v1 = _commit_file(repo, "f.txt", V1, "v1")
    v2 = _commit_file(repo, "f.txt", V2, "v2")
    v3 = _commit_file(repo, "f.txt", V3, "v3")
    gr = Git(str(repo.working_tree_dir))
    yield gr, v1, v2, v3
    gr.clear()


def _mapping(gr: Git, a: str, b: str, filename: str = "f.txt") -> LineMapping:
    mods = [m for m in gr.diff(a, b) if m.filename == filename]
    assert len(mods) == 1
    return mods[0].line_mapping


def test_point_mapping_both_directions(three_versions):
    gr, v1, v2, _ = three_versions
    mapping = _mapping(gr, v1, v2)

    assert mapping.old_lines == 10
    assert mapping.new_lines == 11
    for old, new in OLD_TO_NEW_V2.items():
        assert mapping.old_to_new(old) is new
    for new, old in NEW_TO_OLD_V2.items():
        assert mapping.new_to_old(new) is old


def test_aligned_blocks(three_versions):
    gr, v1, v2, _ = three_versions
    assert _mapping(gr, v1, v2).blocks == BLOCKS_V1_V2


def test_unchanged_regions_have_offsets(three_versions):
    gr, v1, v2, _ = three_versions
    mapping = _mapping(gr, v1, v2)

    # Before the first hunk the line number is unchanged...
    assert mapping.old_to_new(1) == 1
    # ...while every unchanged line after the two insertions is
    # shifted by 2 in the new file (this cannot be derived from
    # diff_parsed, which only lists added/deleted lines).
    assert mapping.old_to_new(2) == 4
    assert mapping.old_to_new(10) == 11
    # The tail after the last hunk is aligned too.
    assert mapping.old_to_new(10) == mapping.new_lines


def test_map_old_range(three_versions):
    gr, v1, v2, _ = three_versions
    mapping = _mapping(gr, v1, v2)

    # A single unchanged line maps to a single range.
    assert mapping.map_old_range(2, 2) == [(4, 4)]
    # A range covering a replacement is split around it.
    assert mapping.map_old_range(2, 6) == [(4, 5), (7, 8)]
    # Contiguous new lines are merged even when old lines were
    # deleted in between (old 6 -> new 8 and old 8 -> new 9).
    assert mapping.map_old_range(1, 10) == [(1, 1), (4, 5), (7, 11)]
    # Ranges entirely inside a changed region map to nothing.
    assert mapping.map_old_range(4, 4) == []
    assert mapping.map_old_range(7, 7) == []


def test_map_new_range(three_versions):
    gr, v1, v2, _ = three_versions
    mapping = _mapping(gr, v1, v2)

    assert mapping.map_new_range(4, 5) == [(2, 3)]
    assert mapping.map_new_range(1, 11) == [(1, 3), (5, 6), (8, 10)]
    assert mapping.map_new_range(6, 6) == []


def test_compose_matches_direct_diff(three_versions):
    gr, v1, v2, v3 = three_versions
    first = _mapping(gr, v1, v2)
    second = _mapping(gr, v2, v3)
    direct = _mapping(gr, v1, v3)

    composed = first.compose(second)

    expected = {
        1: 2, 2: 5, 3: 6, 4: None, 5: 8,
        6: 9, 7: None, 8: 10, 9: None, 10: 12,
    }
    for old, new in expected.items():
        assert composed.old_to_new(old) is new
        assert direct.old_to_new(old) is new

    expected_reverse = {
        1: None, 2: 1, 3: None, 4: None, 5: 2, 6: 3,
        7: None, 8: 5, 9: 6, 10: 8, 11: None, 12: 10,
    }
    for new, old in expected_reverse.items():
        assert composed.new_to_old(new) is old
        assert direct.new_to_old(new) is old

    assert composed.old_lines == 10
    assert composed.new_lines == 12
    assert composed.blocks == direct.blocks
    assert composed.map_old_range(1, 10) == direct.map_old_range(1, 10)
    assert composed.map_new_range(1, 12) == direct.map_new_range(1, 12)
    assert composed.map_old_range(2, 9) == direct.map_old_range(2, 9)


def test_compose_order(three_versions):
    # The mappings must be chained in chronological order: v1->v2
    # followed by a mapping whose old version has a different length.
    gr, v1, v2, v3 = three_versions
    first = _mapping(gr, v1, v2)
    third = _mapping(gr, v1, v3)
    with pytest.raises(ValueError):
        first.compose(third)


def test_inserted_then_deleted_line_disappears(tmp_path: Path):
    # A line that only exists in the intermediate version leaves no
    # trace in the composed mapping: v1 and v3 are byte-identical.
    repo = Repo.init(str(tmp_path / "repo"))
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@test.com")
        cw.set_value("user", "name", "Test")
    a = _commit_file(repo, "f.txt", "a\nb\nc\n", "1")
    b = _commit_file(repo, "f.txt", "a\nX\nb\nc\n", "2")
    c = _commit_file(repo, "f.txt", "a\nb\nc\n", "3")
    gr = Git(str(repo.working_tree_dir))
    try:
        composed = _mapping(gr, a, b).compose(_mapping(gr, b, c))
        assert composed.old_lines == 3
        assert composed.new_lines == 3
        assert composed.old_to_new(1) == 1
        assert composed.old_to_new(2) == 2
        assert composed.old_to_new(3) == 3
        assert composed.map_old_range(1, 3) == [(1, 3)]
        assert all(
            block.type is not AlignmentType.INSERT
            for block in composed.blocks
        )
    finally:
        gr.clear()


def test_deleted_then_readded_line_is_not_the_same_line(tmp_path: Path):
    # Git alignments do not track line identity across versions: a
    # line deleted in v2 and re-added with the same content in v3 is
    # reported as deleted+inserted by the composition, even though the
    # direct diff would see identical content.
    repo = Repo.init(str(tmp_path / "repo"))
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@test.com")
        cw.set_value("user", "name", "Test")
    a = _commit_file(repo, "f.txt", "a\nb\nc\n", "1")
    b = _commit_file(repo, "f.txt", "a\nc\n", "2")
    c = _commit_file(repo, "f.txt", "a\nb\nc\n", "3")
    gr = Git(str(repo.working_tree_dir))
    try:
        composed = _mapping(gr, a, b).compose(_mapping(gr, b, c))
        assert composed.old_to_new(1) == 1
        assert composed.old_to_new(2) is None
        assert composed.old_to_new(3) == 3
        assert composed.new_to_old(2) is None
        assert composed.map_old_range(1, 3) == [(1, 1), (3, 3)]
    finally:
        gr.clear()


def test_added_file(tmp_path: Path):
    repo = Repo.init(str(tmp_path / "repo"))
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@test.com")
        cw.set_value("user", "name", "Test")
    v1 = _commit_file(repo, "new.txt", V1, "add")
    gr = Git(str(repo.working_tree_dir))
    try:
        mod = gr.get_commit(v1).modified_files[0]
        mapping = mod.line_mapping
        assert mod.change_type.name == "ADD"
        assert mapping.old_lines == 0
        assert mapping.new_lines == 10
        assert mapping.blocks == [
            DiffBlock(AlignmentType.INSERT, None, None, 1, 10)
        ]
        assert all(mapping.new_to_old(i) is None for i in range(1, 11))
        assert mapping.map_new_range(1, 10) == []
        with pytest.raises(ValueError):
            mapping.old_to_new(1)
        with pytest.raises(ValueError):
            mapping.map_old_range(1, 1)
    finally:
        gr.clear()


def test_deleted_file(tmp_path: Path):
    repo = Repo.init(str(tmp_path / "repo"))
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@test.com")
        cw.set_value("user", "name", "Test")
    _commit_file(repo, "f.txt", V1, "add")
    repo.index.remove(["f.txt"])
    v2 = repo.index.commit("delete").hexsha
    gr = Git(str(repo.working_tree_dir))
    try:
        mod = gr.get_commit(v2).modified_files[0]
        mapping = mod.line_mapping
        assert mod.change_type.name == "DELETE"
        assert mapping.old_lines == 10
        assert mapping.new_lines == 0
        assert mapping.blocks == [
            DiffBlock(AlignmentType.DELETE, 1, 10, None, None)
        ]
        assert all(mapping.old_to_new(i) is None for i in range(1, 11))
        assert mapping.map_old_range(1, 10) == []
        with pytest.raises(ValueError):
            mapping.new_to_old(1)
    finally:
        gr.clear()


def test_empty_file_added_and_deleted(tmp_path: Path):
    repo = Repo.init(str(tmp_path / "repo"))
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@test.com")
        cw.set_value("user", "name", "Test")
    v1 = _commit_file(repo, "empty.txt", "", "add empty")
    repo.index.remove(["empty.txt"])
    v2 = repo.index.commit("delete empty").hexsha
    gr = Git(str(repo.working_tree_dir))
    try:
        added = gr.get_commit(v1).modified_files[0].line_mapping
        deleted = gr.get_commit(v2).modified_files[0].line_mapping
        assert added.old_lines == 0
        assert added.new_lines == 0
        assert added.blocks == []
        assert deleted.old_lines == 0
        assert deleted.new_lines == 0
    finally:
        gr.clear()


def test_binary_file(tmp_path: Path):
    repo = Repo.init(str(tmp_path / "repo"))
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@test.com")
        cw.set_value("user", "name", "Test")
    path = Path(str(repo.working_tree_dir)) / "blob.dat"
    path.write_bytes(b"\x00\x01\x02\xff")
    repo.index.add(["blob.dat"])
    v1 = repo.index.commit("bin1").hexsha
    path.write_bytes(b"\x00\x01\x03\xff\xfe")
    repo.index.add(["blob.dat"])
    v2 = repo.index.commit("bin2").hexsha
    gr = Git(str(repo.working_tree_dir))
    try:
        mapping = _mapping(gr, v1, v2, "blob.dat")
        assert mapping.blocks == []
        assert mapping.old_to_new(1) is None
        assert mapping.new_to_old(1) is None
        assert mapping.map_old_range(1, 1) == []
    finally:
        gr.clear()


def test_pure_rename_is_identity(tmp_path: Path):
    repo = Repo.init(str(tmp_path / "repo"))
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@test.com")
        cw.set_value("user", "name", "Test")
    _commit_file(repo, "orig.txt", V1, "add")
    repo.index.move(["orig.txt", "moved.txt"])
    v2 = repo.index.commit("rename").hexsha
    gr = Git(str(repo.working_tree_dir))
    try:
        mod = gr.get_commit(v2).modified_files[0]
        mapping = mod.line_mapping
        assert mod.change_type.name == "RENAME"
        # GitPython does not expose the blobs of a 100% rename...
        assert mapping.old_lines is None
        assert mapping.new_lines is None
        # ...but the content is identical, so every line maps to
        # itself.
        for line in (1, 4, 10):
            assert mapping.old_to_new(line) == line
            assert mapping.new_to_old(line) == line
        assert mapping.map_old_range(2, 8) == [(2, 8)]
        with pytest.raises(ValueError):
            mapping.old_to_new(0)
    finally:
        gr.clear()


def test_compose_through_pure_rename(tmp_path: Path):
    repo = Repo.init(str(tmp_path / "repo"))
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@test.com")
        cw.set_value("user", "name", "Test")
    v1 = _commit_file(repo, "orig.txt", V2, "add")
    repo.index.move(["orig.txt", "moved.txt"])
    v2 = repo.index.commit("rename").hexsha
    v3 = _commit_file(repo, "moved.txt", V3, "edit after rename")
    gr = Git(str(repo.working_tree_dir))
    try:
        ren = _mapping(gr, v1, v2, "moved.txt")
        edit = _mapping(gr, v2, v3, "moved.txt")
        composed = ren.compose(edit)
        assert composed.old_lines is None
        assert composed.new_lines == 12
        # The rename changes nothing, then the edit shifts lines:
        # old line 1 (L1) ends up at new line 2.
        assert composed.old_to_new(1) == 2
        assert composed.map_old_range(1, 1) == [(2, 2)]
    finally:
        gr.clear()


def test_rename_with_modification(tmp_path: Path):
    repo = Repo.init(str(tmp_path / "repo"))
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@test.com")
        cw.set_value("user", "name", "Test")
    content = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\n"
    _commit_file(repo, "orig.txt", content, "add")
    repo.index.move(["orig.txt", "moved.txt"])
    Path(str(repo.working_tree_dir), "moved.txt").write_text(
        "line1\nline2\nCHANGED\nline4\nline5\nline6\nline7\nline8\n"
    )
    repo.index.add(["moved.txt"])
    v2 = repo.index.commit("rename and modify").hexsha
    gr = Git(str(repo.working_tree_dir))
    try:
        mod = gr.get_commit(v2).modified_files[0]
        mapping = mod.line_mapping
        assert mod.change_type.name == "RENAME"
        assert mapping.old_lines == 8
        assert mapping.new_lines == 8
        assert mapping.old_to_new(3) is None
        assert mapping.new_to_old(3) is None
        assert mapping.old_to_new(4) == 4
        assert mapping.old_to_new(8) == 8
        assert mapping.map_old_range(1, 8) == [(1, 2), (4, 8)]
    finally:
        gr.clear()


def test_invalid_arguments(three_versions):
    gr, v1, v2, _ = three_versions
    mapping = _mapping(gr, v1, v2)

    for bad in (0, -1, 11, True, 1.0, "1"):
        with pytest.raises(ValueError):
            mapping.old_to_new(bad)
    for bad in (0, -1, 12, False, 1.0):
        with pytest.raises(ValueError):
            mapping.new_to_old(bad)

    for start, end in ((0, 1), (11, 11), (5, 4), (1, 0)):
        with pytest.raises(ValueError):
            mapping.map_old_range(start, end)
    for start, end in ((0, 1), (12, 12), (5, 4)):
        with pytest.raises(ValueError):
            mapping.map_new_range(start, end)


def test_line_mapping_is_cached(three_versions):
    gr, v1, v2, _ = three_versions
    mod = [m for m in gr.diff(v1, v2) if m.filename == "f.txt"][0]
    assert mod.line_mapping is mod.line_mapping


def test_parse_diff_unit():
    diff = (
        "@@ -1,2 +1,2 @@\n"
        " a\n"
        "-b\n"
        "+B\n"
        "@@ -8,3 +8,4 @@\n"
        " context1\n"
        "-old\n"
        "+new1\n"
        "+new2\n"
        " context2\n"
    )
    mapping = LineMapping(diff, 11, 12)

    assert mapping.blocks == [
        DiffBlock(AlignmentType.EQUAL, 1, 1, 1, 1),
        DiffBlock(AlignmentType.DELETE, 2, 2, None, None),
        DiffBlock(AlignmentType.INSERT, None, None, 2, 2),
        # unchanged gap between the two hunks merged with the first
        # context line of the second hunk
        DiffBlock(AlignmentType.EQUAL, 3, 8, 3, 8),
        DiffBlock(AlignmentType.DELETE, 9, 9, None, None),
        DiffBlock(AlignmentType.INSERT, None, None, 9, 10),
        # context merged with the unchanged tail after the last hunk
        DiffBlock(AlignmentType.EQUAL, 10, 11, 11, 12),
    ]
    assert mapping.old_to_new(7) == 7
    assert mapping.old_to_new(9) is None
    assert mapping.old_to_new(11) == 12
    assert mapping.new_to_old(12) == 11


def test_parse_diff_no_newline_marker():
    diff = (
        "@@ -1,3 +1,4 @@\n"
        " a\n"
        "-b\n"
        "+B\n"
        "+c\n"
        " d\n"
        r"\ No newline at end of file"
    )
    mapping = LineMapping(diff, 3, 4)
    assert mapping.blocks == [
        DiffBlock(AlignmentType.EQUAL, 1, 1, 1, 1),
        DiffBlock(AlignmentType.DELETE, 2, 2, None, None),
        DiffBlock(AlignmentType.INSERT, None, None, 2, 3),
        DiffBlock(AlignmentType.EQUAL, 3, 3, 4, 4),
    ]
    assert mapping.old_to_new(3) == 4
    assert mapping.new_to_old(3) is None


@pytest.mark.parametrize(
    "commit,expected_old,expected_new",
    [
        # 5 pure additions: the old file has 640 lines, the new one 645.
        ("f45ee2f8976d5f018a1e4ec83eb4556a3df8b0a5", 640, 645),
        # 12 pure deletions.
        ("147c7ce9f725a0e259d63f0bf4e6c8ac085ff8c8", 645, 633),
    ],
)
def test_offsets_on_real_repository(commit, expected_old, expected_new):
    gr = Git("test-repos/diff")
    try:
        mod = gr.get_commit(commit).modified_files[0]
        mapping = mod.line_mapping
        assert isinstance(mod, ModifiedFile)
        assert mapping.old_lines == expected_old
        assert mapping.new_lines == expected_new

        # The mapping must be a bijection on the context lines: walking
        # it in both directions returns the original line.
        for old in range(1, mapping.old_lines + 1):
            new = mapping.old_to_new(old)
            if new is not None:
                assert mapping.new_to_old(new) == old
        for new in range(1, mapping.new_lines + 1):
            old = mapping.new_to_old(new)
            if old is not None:
                assert mapping.old_to_new(old) == new
    finally:
        gr.clear()


def test_no_newline_repository():
    gr = Git("test-repos/no_newline")
    try:
        mod = gr.get_commit(
            "52a78c1ee5d100528eccba0a3d67371dbd22d898"
        ).modified_files[0]
        mapping = mod.line_mapping
        assert mapping.old_lines == 1
        assert mapping.new_lines == 2
        assert mapping.old_to_new(1) is None
        assert mapping.new_to_old(1) is None
        assert mapping.new_to_old(2) is None
        assert mapping.blocks == [
            DiffBlock(AlignmentType.DELETE, 1, 1, None, None),
            DiffBlock(AlignmentType.INSERT, None, None, 1, 2),
        ]
    finally:
        gr.clear()
