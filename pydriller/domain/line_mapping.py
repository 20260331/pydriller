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
This module contains the classes used to map lines (and ranges of lines)
between two versions of the same file.

The mapping is derived from the unified diff produced by Git: context
lines (the lines prefixed with a space in a hunk) are aligned one to one
between the old and the new version, while lines that only appear on one
side (prefixed with ``-`` or ``+``) have no counterpart. Regions of the
file that are not touched by any hunk are aligned too: this is what
allows the offset of an unchanged line to be computed even when hunks
add or remove lines before it.
"""
import bisect
import re
from enum import Enum
from typing import List, NamedTuple, Optional, Tuple


class AlignmentType(Enum):
    """
    Type of an aligned :class:`DiffBlock`.

    * EQUAL: lines present unchanged in both versions
    * DELETE: lines present only in the old version
    * INSERT: lines present only in the new version
    """

    EQUAL = 1
    DELETE = 2
    INSERT = 3


class DiffBlock(NamedTuple):
    """
    A contiguous aligned region between the old and the new version of a
    file. Line numbers are 1-based and inclusive. The sides that do not
    participate in the alignment are None (e.g. ``new_start`` and
    ``new_end`` for a DELETE block).

    * For an EQUAL block, both ranges have the same length and line
      ``old_start + k`` corresponds to ``new_start + k``.
    * For a DELETE block, only the old range is set.
    * For an INSERT block, only the new range is set.
    """

    type: AlignmentType
    old_start: Optional[int]
    old_end: Optional[int]
    new_start: Optional[int]
    new_end: Optional[int]


_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

_NO_NEWLINE_MARKER = r"\ No newline at end of file"


class LineMapping:
    """
    Bidirectional mapping between the 1-based line numbers of two
    versions of a file, built from the unified diff of the change.

    Only unchanged lines (context lines in the diff) have a definite
    counterpart: the Git alignment does not pair individual deleted
    lines with individual added lines, so lines inside a changed region
    map to None. Ranges are mapped to the maximal contiguous ranges in
    the other version that are covered by unchanged lines.

    Two adjacent mappings can be composed with :func:`compose` to map
    lines across multiple versions (e.g. v1 -> v2 and v2 -> v3 into
    v1 -> v3).
    """

    def __init__(
        self,
        diff: str,
        old_lines: Optional[int],
        new_lines: Optional[int],
    ) -> None:
        """
        Build a line mapping from a unified diff.

        Normally a LineMapping is obtained through
        :attr:`ModifiedFile.line_mapping`, constructing one by hand is
        only useful for testing.

        :param str diff: the unified diff, as returned by
            :attr:`ModifiedFile.diff`
        :param Optional[int] old_lines: total number of lines in the old
            version, or None if it cannot be determined
        :param Optional[int] new_lines: total number of lines in the new
            version, or None if it cannot be determined
        """
        self._old_lines = old_lines
        self._new_lines = new_lines
        self._blocks = self._parse_diff(diff, old_lines, new_lines)

        # Sorted (start, end, counterpart_start) entries for EQUAL blocks,
        # used to answer point and range lookups in logarithmic time.
        self._old_equal: List[Tuple[int, int, int]] = []
        self._new_equal: List[Tuple[int, int, int]] = []
        for block in self._blocks:
            if block.type is not AlignmentType.EQUAL:
                continue
            assert block.old_start is not None and block.old_end is not None
            assert block.new_start is not None and block.new_end is not None
            self._old_equal.append(
                (block.old_start, block.old_end, block.new_start)
            )
            self._new_equal.append(
                (block.new_start, block.new_end, block.old_start)
            )

    @property
    def old_lines(self) -> Optional[int]:
        """
        :return: total number of lines in the old version, or None when
            the length cannot be determined (a 100% rename, for which
            Git does not expose the blobs: the mapping is then the
            identity)
        """
        return self._old_lines

    @property
    def new_lines(self) -> Optional[int]:
        """
        :return: total number of lines in the new version, or None when
            the length cannot be determined (a 100% rename, for which
            Git does not expose the blobs: the mapping is then the
            identity)
        """
        return self._new_lines

    @property
    def blocks(self) -> List[DiffBlock]:
        """
        Return the aligned blocks of the diff, in file order. Adjacent
        blocks of the same type are merged.

        :return: list of :class:`DiffBlock`
        """
        return list(self._blocks)

    @staticmethod
    def _count_lines(content: Optional[bytes]) -> int:
        """
        Count the lines of a blob the same way Git does when numbering
        the lines of a hunk: a file ending with a newline does not have
        an extra empty line after it, and an empty file has 0 lines.
        """
        if not content:
            return 0
        lines = content.count(b"\n")
        if not content.endswith(b"\n"):
            lines += 1
        return lines

    @classmethod
    def _open_identity(cls) -> "LineMapping":
        """
        Build the identity mapping used when the two versions are known
        to be identical but their length is not available (a pure
        rename, where Git does not expose the blobs). Every line maps
        to itself, without any upper-bound check.
        """
        mapping = cls.__new__(cls)
        mapping._old_lines = None
        mapping._new_lines = None
        mapping._blocks = []
        mapping._old_equal = []
        mapping._new_equal = []
        return mapping

    def _is_open_identity(self) -> bool:
        return (
            not self._blocks
            and self._old_lines is None
            and self._new_lines is None
        )

    @classmethod
    def _from_blocks(
        cls,
        blocks: List[DiffBlock],
        old_lines: Optional[int],
        new_lines: Optional[int],
    ) -> "LineMapping":
        mapping = cls.__new__(cls)
        mapping._old_lines = old_lines
        mapping._new_lines = new_lines
        mapping._blocks = blocks
        mapping._old_equal = []
        mapping._new_equal = []
        for block in blocks:
            if block.type is not AlignmentType.EQUAL:
                continue
            assert block.old_start is not None and block.old_end is not None
            assert block.new_start is not None and block.new_end is not None
            mapping._old_equal.append(
                (block.old_start, block.old_end, block.new_start)
            )
            mapping._new_equal.append(
                (block.new_start, block.new_end, block.old_start)
            )
        return mapping

    # ------------------------------------------------------------------ #
    # Diff parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_diff(
        diff: str,
        old_total: Optional[int],
        new_total: Optional[int],
    ) -> List[DiffBlock]:
        """
        Walk every hunk while keeping track of the current old and new
        line number, grouping consecutive lines of the same kind into
        blocks. Unchanged gaps before, between and after hunks are added
        as EQUAL blocks so that offsets outside the hunks are known.
        """
        for raw_line in diff.split("\n"):
            if raw_line.startswith("Binary files"):
                # The content of a binary file cannot be aligned.
                return []

        blocks: List[DiffBlock] = []
        old_cursor = 1
        new_cursor = 1

        lines = diff.replace("\r", "").split("\n")
        index = 0
        while index < len(lines):
            header = _HUNK_HEADER_RE.match(lines[index])
            if header is None:
                index += 1
                continue

            hunk_old_start = int(header.group(1))
            hunk_new_start = int(header.group(3))

            # Unchanged gap between the previous hunk and this one.
            # Starts can be 0 for hunks that add (old side) or delete
            # (new side) the whole file; such hunks have no gap on the
            # missing side.
            gap_old = hunk_old_start - old_cursor
            gap_new = hunk_new_start - new_cursor
            if gap_old >= 0 and gap_new >= 0 and (
                gap_old > 0 or gap_new > 0
            ):
                blocks.append(
                    DiffBlock(
                        AlignmentType.EQUAL,
                        old_cursor,
                        old_cursor + gap_old - 1,
                        new_cursor,
                        new_cursor + gap_new - 1,
                    )
                )
                old_cursor += gap_old
                new_cursor += gap_new

            old_pos = hunk_old_start
            new_pos = hunk_new_start
            run_type: Optional[AlignmentType] = None
            run_old_start = 0
            run_new_start = 0
            run_length = 0

            def flush_run() -> None:
                if run_type is None or run_length == 0:
                    return
                if run_type is AlignmentType.EQUAL:
                    blocks.append(
                        DiffBlock(
                            run_type,
                            run_old_start,
                            run_old_start + run_length - 1,
                            run_new_start,
                            run_new_start + run_length - 1,
                        )
                    )
                elif run_type is AlignmentType.DELETE:
                    blocks.append(
                        DiffBlock(
                            run_type,
                            run_old_start,
                            run_old_start + run_length - 1,
                            None,
                            None,
                        )
                    )
                else:
                    blocks.append(
                        DiffBlock(
                            run_type,
                            None,
                            None,
                            run_new_start,
                            run_new_start + run_length - 1,
                        )
                    )

            index += 1
            while index < len(lines):
                line = lines[index]
                if _HUNK_HEADER_RE.match(line):
                    break
                index += 1

                if line == "" or line == _NO_NEWLINE_MARKER:
                    continue
                if line.startswith("Binary files"):
                    return []

                if line.startswith(" "):
                    line_type = AlignmentType.EQUAL
                elif line.startswith("-"):
                    line_type = AlignmentType.DELETE
                elif line.startswith("+"):
                    line_type = AlignmentType.INSERT
                else:
                    # File headers ("diff --git", "index", ...) are not
                    # part of the body of a hunk; ignore anything we do
                    # not recognize.
                    continue

                if line_type is not run_type:
                    flush_run()
                    run_type = line_type
                    run_old_start = old_pos
                    run_new_start = new_pos
                    run_length = 0

                run_length += 1
                if line_type is AlignmentType.EQUAL:
                    old_pos += 1
                    new_pos += 1
                elif line_type is AlignmentType.DELETE:
                    old_pos += 1
                else:
                    new_pos += 1

            flush_run()
            old_cursor = old_pos
            new_cursor = new_pos

        # Unchanged tail after the last hunk (or the whole file when the
        # diff has no hunk, e.g. a rename with identical content).
        if old_total is not None and new_total is not None:
            tail_old = old_total - old_cursor + 1
            tail_new = new_total - new_cursor + 1
            if tail_old > 0 and tail_new > 0 and tail_old == tail_new:
                blocks.append(
                    DiffBlock(
                        AlignmentType.EQUAL,
                        old_cursor,
                        old_total,
                        new_cursor,
                        new_total,
                    )
                )

        return LineMapping._merge_adjacent(blocks)

    @staticmethod
    def _merge_adjacent(blocks: List[DiffBlock]) -> List[DiffBlock]:
        merged: List[DiffBlock] = []
        for block in blocks:
            if not merged:
                merged.append(block)
                continue
            previous = merged[-1]
            if previous.type is not block.type:
                merged.append(block)
                continue

            contiguous = False
            if block.type is AlignmentType.EQUAL:
                assert (
                    previous.old_end is not None
                    and block.old_start is not None
                    and previous.new_end is not None
                    and block.new_start is not None
                )
                contiguous = (
                    previous.old_end == block.old_start - 1
                    and previous.new_end == block.new_start - 1
                )
            elif block.type is AlignmentType.DELETE:
                assert (
                    previous.old_end is not None
                    and block.old_start is not None
                )
                contiguous = previous.old_end == block.old_start - 1
            else:
                assert (
                    previous.new_end is not None
                    and block.new_start is not None
                )
                contiguous = previous.new_end == block.new_start - 1

            if not contiguous:
                merged.append(block)
                continue

            merged[-1] = DiffBlock(
                block.type,
                previous.old_start,
                block.old_end,
                previous.new_start,
                block.new_end,
            )
        return merged

    # ------------------------------------------------------------------ #
    # Point lookups
    # ------------------------------------------------------------------ #

    def old_to_new(self, old_line: int) -> Optional[int]:
        """
        Return the line in the new version that corresponds to the
        given line of the old version.

        :param int old_line: 1-based line number in the old version
        :return: the corresponding 1-based line number in the new
            version, or None if the line was deleted or otherwise
            belongs to a changed region
        :raises ValueError: if the line number is not valid for the old
            version
        """
        self._validate_line(old_line, self._old_lines, "old")
        if self._is_open_identity():
            return old_line
        return self._lookup(old_line, self._old_equal)

    def new_to_old(self, new_line: int) -> Optional[int]:
        """
        Return the line in the old version that corresponds to the
        given line of the new version.

        :param int new_line: 1-based line number in the new version
        :return: the corresponding 1-based line number in the old
            version, or None if the line was added or otherwise belongs
            to a changed region
        :raises ValueError: if the line number is not valid for the new
            version
        """
        self._validate_line(new_line, self._new_lines, "new")
        if self._is_open_identity():
            return new_line
        return self._lookup(new_line, self._new_equal)

    @staticmethod
    def _lookup(
        line: int, equal: List[Tuple[int, int, int]]
    ) -> Optional[int]:
        if not equal:
            return None
        index = bisect.bisect_right(equal, (line, float("inf"), float("inf"))) - 1
        if index < 0:
            return None
        start, end, counterpart_start = equal[index]
        if start <= line <= end:
            return counterpart_start + (line - start)
        return None

    @staticmethod
    def _validate_line(
        line: int, total: Optional[int], side: str
    ) -> None:
        if not isinstance(line, int) or isinstance(line, bool):
            raise ValueError(f"The {side} line number must be an integer")
        if line < 1 or (total is not None and line > total):
            raise ValueError(
                f"The {side} file has {total} lines, line {line} does not exist"
            )

    # ------------------------------------------------------------------ #
    # Range lookups
    # ------------------------------------------------------------------ #

    def map_old_range(
        self, start: int, end: int
    ) -> List[Tuple[int, int]]:
        """
        Map an inclusive 1-based range of the old version to the
        maximal contiguous inclusive ranges of the new version covered
        by unchanged lines. Lines deleted in between are skipped, so a
        range can map to zero, one or multiple ranges.

        :param int start: first line of the range in the old version
        :param int end: last line of the range in the old version
        :return: sorted list of (start, end) ranges in the new version
        :raises ValueError: if the range is not valid for the old
            version
        """
        self._validate_range(start, end, self._old_lines, "old")
        if self._is_open_identity():
            return [(start, end)]
        return self._map_range(start, end, self._old_equal)

    def map_new_range(
        self, start: int, end: int
    ) -> List[Tuple[int, int]]:
        """
        Map an inclusive 1-based range of the new version to the
        maximal contiguous inclusive ranges of the old version covered
        by unchanged lines. Lines added in between are skipped, so a
        range can map to zero, one or multiple ranges.

        :param int start: first line of the range in the new version
        :param int end: last line of the range in the new version
        :return: sorted list of (start, end) ranges in the old version
        :raises ValueError: if the range is not valid for the new
            version
        """
        self._validate_range(start, end, self._new_lines, "new")
        if self._is_open_identity():
            return [(start, end)]
        return self._map_range(start, end, self._new_equal)

    @staticmethod
    def _map_range(
        start: int,
        end: int,
        equal: List[Tuple[int, int, int]],
    ) -> List[Tuple[int, int]]:
        mapped: List[Tuple[int, int]] = []
        first = bisect.bisect_right(
            equal, (start, float("inf"), float("inf"))
        ) - 1
        for index in range(max(first, 0), len(equal)):
            block_start, block_end, counterpart_start = equal[index]
            if block_start > end:
                break
            overlap_start = max(start, block_start)
            overlap_end = min(end, block_end)
            if overlap_start > overlap_end:
                continue
            mapped_range = (
                counterpart_start + (overlap_start - block_start),
                counterpart_start + (overlap_end - block_start),
            )
            if mapped and mapped_range[0] <= mapped[-1][1] + 1:
                mapped[-1] = (mapped[-1][0], mapped_range[1])
            else:
                mapped.append(mapped_range)
        return mapped

    @staticmethod
    def _validate_range(
        start: int, end: int, total: Optional[int], side: str
    ) -> None:
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
        ):
            raise ValueError("Range boundaries must be integers")
        if start < 1 or end < start or (total is not None and end > total):
            raise ValueError(
                f"Not a valid range of the {side} file (which has "
                f"{total} lines): [{start}, {end}]"
            )

    # ------------------------------------------------------------------ #
    # Composition
    # ------------------------------------------------------------------ #

    def compose(self, other: "LineMapping") -> "LineMapping":
        """
        Compose this mapping with the mapping of the next adjacent
        versions. If this mapping goes from v1 to v2 and ``other`` goes
        from v2 to v3, the returned mapping goes from v1 to v3: a line
        has a counterpart across the composition only if it survived
        unchanged through both changes.

        :param LineMapping other: mapping from v2 to v3
        :return: mapping from v1 to v3
        :raises ValueError: if the intermediate versions do not have
            the same number of lines, i.e. the mappings cannot be
            chained
        """
        if (
            self._new_lines is not None
            and other._old_lines is not None
            and self._new_lines != other._old_lines
        ):
            raise ValueError(
                "Cannot compose the mappings: the new version of the "
                f"first mapping has {self._new_lines} lines while the "
                f"old version of the second has {other._old_lines}"
            )

        # An open identity (pure rename whose length Git does not
        # expose) leaves every coordinate untouched, so chaining it
        # with the other mapping just propagates the unknown length.
        if self._is_open_identity():
            return LineMapping._from_blocks(
                list(other._blocks), None, other._new_lines
            )
        if other._is_open_identity():
            return LineMapping._from_blocks(
                list(self._blocks), self._old_lines, None
            )

        composed: List[DiffBlock] = []

        def emit(
            block_type: AlignmentType,
            old_start: Optional[int],
            old_end: Optional[int],
            new_start: Optional[int],
            new_end: Optional[int],
        ) -> None:
            block = DiffBlock(
                block_type, old_start, old_end, new_start, new_end
            )
            if composed and composed[-1].type is block_type:
                previous = composed[-1]
                contiguous = False
                if block_type is AlignmentType.EQUAL:
                    assert (
                        previous.old_end is not None
                        and old_start is not None
                        and previous.new_end is not None
                        and new_start is not None
                    )
                    contiguous = (
                        previous.old_end == old_start - 1
                        and previous.new_end == new_start - 1
                    )
                elif block_type is AlignmentType.DELETE:
                    assert (
                        previous.old_end is not None
                        and old_start is not None
                    )
                    contiguous = previous.old_end == old_start - 1
                else:
                    assert (
                        previous.new_end is not None
                        and new_start is not None
                    )
                    contiguous = previous.new_end == new_start - 1
                if contiguous:
                    composed[-1] = DiffBlock(
                        block_type,
                        previous.old_start,
                        old_end,
                        previous.new_start,
                        new_end,
                    )
                    return
            composed.append(block)

        # Sweep the two block sequences along the v2 coordinates.
        # old_pos/new_pos are the next unassigned 1-based positions in
        # v1/v3, offsets remember how many lines of a block were
        # already consumed when a block is only partially advanced.
        old_pos = new_pos = 1
        index_self = index_other = 0
        offset_self = offset_other = 0

        def self_left() -> int:
            block = self._blocks[index_self]
            if block.type is AlignmentType.INSERT:
                assert (
                    block.new_start is not None
                    and block.new_end is not None
                )
                return block.new_end - block.new_start + 1 - offset_self
            assert block.old_start is not None and block.old_end is not None
            return block.old_end - block.old_start + 1 - offset_self

        def other_left() -> int:
            block = other._blocks[index_other]
            if block.type is AlignmentType.INSERT:
                assert (
                    block.new_start is not None
                    and block.new_end is not None
                )
                return block.new_end - block.new_start + 1 - offset_other
            assert block.old_start is not None and block.old_end is not None
            return block.old_end - block.old_start + 1 - offset_other

        while (
            index_self < len(self._blocks)
            or index_other < len(other._blocks)
        ):
            block_self = (
                self._blocks[index_self]
                if index_self < len(self._blocks)
                else None
            )
            block_other = (
                other._blocks[index_other]
                if index_other < len(other._blocks)
                else None
            )

            # v1-only lines (deleted before v2) never reach v3.
            if (
                block_self is not None
                and block_self.type is AlignmentType.DELETE
            ):
                length = self_left()
                emit(
                    AlignmentType.DELETE,
                    old_pos,
                    old_pos + length - 1,
                    None,
                    None,
                )
                old_pos += length
                index_self += 1
                offset_self = 0
                continue

            # v3-only lines (added after v2) have no v1 counterpart.
            if (
                block_other is not None
                and block_other.type is AlignmentType.INSERT
            ):
                length = other_left()
                emit(
                    AlignmentType.INSERT,
                    None,
                    None,
                    new_pos,
                    new_pos + length - 1,
                )
                new_pos += length
                index_other += 1
                offset_other = 0
                continue

            if block_self is None:
                # Remaining v2 lines have no v1 counterpart: surviving
                # ones are insertions in v3, deleted ones disappear.
                assert block_other is not None
                length = other_left()
                if block_other.type is AlignmentType.EQUAL:
                    emit(
                        AlignmentType.INSERT,
                        None,
                        None,
                        new_pos,
                        new_pos + length - 1,
                    )
                    new_pos += length
                index_other += 1
                offset_other = 0
                continue

            if block_other is None:
                # v2 continues past the known v3 world: v1 lines that
                # reached v2 are lost, v2-only lines disappear.
                length = self_left()
                if block_self.type is AlignmentType.EQUAL:
                    emit(
                        AlignmentType.DELETE,
                        old_pos,
                        old_pos + length - 1,
                        None,
                        None,
                    )
                    old_pos += length
                index_self += 1
                offset_self = 0
                continue

            if block_self.type is AlignmentType.INSERT:
                # The first block only advances v2 coordinates.
                step = self_left()
                if block_other.type is AlignmentType.DELETE:
                    # Present in v2 only, then removed before v3: the
                    # lines belong to neither endpoint.
                    step = min(step, other_left())
                else:
                    # EQUAL in the second mapping: the lines survive in
                    # v3 but still have no v1 counterpart.
                    emit(
                        AlignmentType.INSERT,
                        None,
                        None,
                        new_pos,
                        new_pos + step - 1,
                    )
                    new_pos += step
            elif block_other.type is AlignmentType.DELETE:
                # The v1 lines reached v2 unchanged but are removed in
                # v3: they are deleted across the composition.
                step = min(self_left(), other_left())
                emit(
                    AlignmentType.DELETE,
                    old_pos,
                    old_pos + step - 1,
                    None,
                    None,
                )
                old_pos += step
            else:
                # Both blocks are EQUAL: the lines survive unchanged
                # through the whole chain.
                step = min(self_left(), other_left())
                emit(
                    AlignmentType.EQUAL,
                    old_pos,
                    old_pos + step - 1,
                    new_pos,
                    new_pos + step - 1,
                )
                old_pos += step
                new_pos += step

            if step == self_left():
                index_self += 1
                offset_self = 0
            else:
                offset_self += step
            if step == other_left():
                index_other += 1
                offset_other = 0
            else:
                offset_other += step

        return LineMapping._from_blocks(
            composed, self._old_lines, other._new_lines
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LineMapping):
            return NotImplemented
        return (
            self._old_lines == other._old_lines
            and self._new_lines == other._new_lines
            and self._blocks == other._blocks
        )

    def __repr__(self) -> str:
        return (
            f"LineMapping(old_lines={self._old_lines}, "
            f"new_lines={self._new_lines}, blocks={self._blocks})"
        )
