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
Fixed selection plans and resumable cursors for long-running traversals.

When a traversal takes a long time, the repository history may change while
it is running (e.g. branches move) and workers may acknowledge commits out of
order. A simple "last processed hash" saved on the side is not enough to
resume reliably in that scenario.

This module provides two artifacts:

* :class:`SelectionPlan` is an immutable snapshot of the *ordered* commits
  selected at planning time. Once written, it never changes; resuming a
  traversal always replays the same hashes in the same order instead of
  re-running ``git rev-list`` against a possibly different history.
* :class:`ResumeCursor` records how far the traversal got. It can only move
  forward and only for commits that were actually delivered. Commits
  acknowledged out of order are remembered but do not advance the cursor
  until the gap before them is filled.

Both artifacts are persisted as JSON (atomic writes) and carry a checksum
so that corruption, a cursor pointing at a different plan, or a plan that
does not match the repository anymore are detected instead of silently
switching to the new history.
"""

import hashlib
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


class SelectionPlanError(Exception):
    """
    Raised when a selection plan or a resume cursor cannot be loaded,
    verified, or advanced consistently. The error is always fatal: the
    caller must not fall back to a freshly computed selection.
    """


def _checksum(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _atomic_write_json(path: Union[str, os.PathLike], data: Dict[str, Any]) -> None:
    """
    Write JSON to ``path`` atomically (temp file + os.replace) so a crash
    while saving can never leave a half-written artifact.
    """
    target = Path(path)
    directory = target.parent if str(target.parent) else Path(".")
    fd, tmp_name = tempfile.mkstemp(
        prefix="." + target.name + ".", suffix=".tmp", dir=str(directory)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(data, tmp_file, indent=2, sort_keys=True)
            tmp_file.write("\n")
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_json(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as plan_file:
            data = json.load(plan_file)
    except (OSError, ValueError) as exc:
        raise SelectionPlanError(f"Cannot read checkpoint file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SelectionPlanError(f"Checkpoint file {path} does not contain a JSON object")
    return data


class SelectionPlan:
    """
    Immutable, ordered snapshot of the commits selected for a traversal.

    The plan is the single source of truth while a traversal is running:
    commits are replayed from ``commits`` (oldest to newest, in the order
    decided at planning time), regardless of branches moving or new commits
    landing in the repository afterwards.

    :param commit_hashes: selected commit hashes, in traversal order.
        Duplicates are not allowed because the cursor indexes this list.
    :param repo: free-form repository identifier (usually the resolved
        repository path); informational only.
    :param head: HEAD commit at planning time, if available.
    :param tagged_commits: frozen snapshot of the tagged commits when the
        ``only_releases`` filter is used (otherwise ``None``).
    :param filepath_commits: frozen snapshot of the commits that modified
        the requested filepath (otherwise ``None``).
    :param created_at: planning timestamp (ISO format); mostly useful in
        the serialized file.
    """

    def __init__(
        self,
        commit_hashes: Sequence[str],
        repo: Optional[str] = None,
        head: Optional[str] = None,
        tagged_commits: Optional[Iterable[str]] = None,
        filepath_commits: Optional[Iterable[str]] = None,
        created_at: Optional[str] = None,
    ) -> None:
        commits = list(commit_hashes)
        duplicates = self._find_duplicates(commits)
        if duplicates:
            raise SelectionPlanError(
                "A selection plan cannot contain duplicate commits, found the "
                f"same commit more than once (first duplicate: {duplicates[0]})"
            )

        self._commits: Tuple[str, ...] = tuple(commits)
        self.repo = repo
        self.head = head
        self.tagged_commits: Optional[FrozenSet[str]] = (
            None if tagged_commits is None else frozenset(tagged_commits)
        )
        self.filepath_commits: Optional[FrozenSet[str]] = (
            None if filepath_commits is None else frozenset(filepath_commits)
        )
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.plan_id = self._compute_plan_id()

    @staticmethod
    def _find_duplicates(commits: List[str]) -> List[str]:
        seen: Set[str] = set()
        duplicates = []
        for commit_hash in commits:
            if commit_hash in seen and commit_hash not in duplicates:
                duplicates.append(commit_hash)
            seen.add(commit_hash)
        return duplicates

    @property
    def commits(self) -> Tuple[str, ...]:
        """Selected commits in traversal order (read-only)."""
        return self._commits

    def _identity_payload(self) -> Dict[str, Any]:
        return {
            "repo": self.repo,
            "head": self.head,
            "commits": list(self._commits),
            "tagged_commits": (
                sorted(self.tagged_commits) if self.tagged_commits is not None else None
            ),
            "filepath_commits": (
                sorted(self.filepath_commits)
                if self.filepath_commits is not None
                else None
            ),
        }

    def _compute_plan_id(self) -> str:
        return _checksum("selection-plan:" + _canonical_json(self._identity_payload()))

    def __len__(self) -> int:
        return len(self._commits)

    def __iter__(self) -> Iterator[str]:
        return iter(self._commits)

    def __getitem__(self, index: int) -> str:
        return self._commits[index]

    def index_of(self, commit_hash: str) -> int:
        """Index of a planned commit (raises if the commit is not planned)."""
        try:
            return self._commits.index(commit_hash)
        except ValueError as exc:
            raise SelectionPlanError(
                f"Commit {commit_hash} is not part of selection plan {self.plan_id}"
            ) from exc

    def missing_commits(self, available: Iterable[str]) -> List[str]:
        """
        Return the planned commits that are not in ``available``.

        :param available: hashes known to exist in the repository object
            store (e.g. output of ``git cat-file --batch-check``).
        """
        available_set = set(available)
        return [commit_hash for commit_hash in self._commits
                if commit_hash not in available_set]

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "version": _SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "created_at": self.created_at,
        }
        data.update(self._identity_payload())
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelectionPlan":
        if data.get("version") != _SCHEMA_VERSION:
            raise SelectionPlanError(
                f"Unsupported selection plan version: {data.get('version')!r}"
            )
        for required in ("plan_id", "commits"):
            if required not in data:
                raise SelectionPlanError(
                    f"Selection plan is missing the '{required}' field"
                )

        plan = cls(
            data["commits"],
            repo=data.get("repo"),
            head=data.get("head"),
            tagged_commits=data.get("tagged_commits"),
            filepath_commits=data.get("filepath_commits"),
            created_at=data.get("created_at"),
        )
        stored_id = data["plan_id"]
        if not isinstance(stored_id, str) or plan.plan_id != stored_id:
            raise SelectionPlanError(
                "Selection plan checksum verification failed: the file was "
                "corrupted or modified. Rebuild a new plan instead of "
                "resuming this one."
            )
        return plan

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Atomically persist the plan as JSON to ``path``."""
        _atomic_write_json(path, self.to_dict())
        logger.info("Saved selection plan %s with %d commits to %s",
                    self.plan_id[:12], len(self._commits), path)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "SelectionPlan":
        """Load and verify (checksum + schema) a plan from ``path``."""
        plan = cls.from_dict(_read_json(path))
        logger.info("Loaded selection plan %s with %d commits from %s",
                    plan.plan_id[:12], len(plan), path)
        return plan


class ResumeCursor:
    """
    Monotonic, verifiable progress cursor bound to a :class:`SelectionPlan`.

    The cursor points at the first commit of the plan that has not been
    delivered yet (``position``). Commits acknowledged out of order are
    stored in ``delivered`` but do not move the cursor until every commit
    before them has been delivered as well.

    The cursor cannot:

    * move backwards (acknowledging an already delivered commit raises);
    * skip commits (acknowledging a future commit only parks it);
    * acknowledge commits that are not part of the plan;
    * be used with a different plan (the ``plan_id`` is checked).
    """

    def __init__(
        self,
        plan: SelectionPlan,
        position: int = 0,
        delivered: Optional[Iterable[str]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._path: Optional[Path] = None
        self._plan = plan
        self.plan_id = plan.plan_id
        self._position = position
        self._delivered: Set[str] = set(delivered) if delivered else set()
        self._validate_state()

    @property
    def plan(self) -> SelectionPlan:
        return self._plan

    @property
    def position(self) -> int:
        """Index of the first commit not delivered yet (contiguous prefix)."""
        return self._position

    @property
    def delivered(self) -> FrozenSet[str]:
        """Commits delivered beyond ``position`` (out of order), read-only."""
        return frozenset(self._delivered)

    @property
    def is_complete(self) -> bool:
        """True when every planned commit has been delivered in order."""
        return self._position >= len(self._plan)

    def is_pending(self, index: int, commit_hash: str) -> bool:
        """
        Whether the planned item at ``index`` still has to be replayed,
        i.e. it is before the contiguous delivered prefix or parked as an
        out-of-order delivery.
        """
        with self._lock:
            if index < self._position:
                return False
            return commit_hash not in self._delivered

    def pending_commits(self) -> List[str]:
        """
        Planned commits that still have to be delivered, in plan order.
        Commits parked as out-of-order deliveries are excluded because they
        must not be replayed.
        """
        with self._lock:
            return [
                commit_hash
                for index, commit_hash in enumerate(self._plan.commits)
                if index >= self._position and commit_hash not in self._delivered
            ]

    def _validate_state(self) -> None:
        if not isinstance(self._position, int) or isinstance(self._position, bool):
            raise SelectionPlanError("Cursor position must be an integer")
        if not 0 <= self._position <= len(self._plan):
            raise SelectionPlanError(
                f"Cursor position {self._position} is outside the plan "
                f"(plan has {len(self._plan)} commits)"
            )
        ahead = set(self._plan.commits[self._position:])
        unknown = [h for h in self._delivered if h not in ahead]
        if unknown:
            raise SelectionPlanError(
                "Cursor records delivered commits that are not pending in the "
                f"plan (first offender: {unknown[0]})"
            )

    def mark_delivered(self, commit_hash: str) -> bool:
        """
        Record that ``commit_hash`` has been delivered.

        If it is the next expected commit, the cursor advances as far as
        possible through the contiguous delivered prefix (swallowing commits
        previously acknowledged out of order). If it is a later planned
        commit, it is parked as an out-of-order delivery and the cursor does
        not move.

        :return: True if the contiguous ``position`` moved, False if the
            commit was parked out of order.
        :raises SelectionPlanError: if the commit is unknown to the plan,
            was already delivered (backward move), or the plan is complete.
        """
        with self._lock:
            if self.is_complete:
                raise SelectionPlanError(
                    f"Cannot mark {commit_hash} delivered: the plan is already complete"
                )

            planned = self._plan.commits
            if self._position < len(planned) and commit_hash == planned[self._position]:
                self._position += 1
                advanced = True
                # Collapse the out-of-order tail that is now contiguous.
                while (
                    self._position < len(planned)
                    and planned[self._position] in self._delivered
                ):
                    self._delivered.discard(planned[self._position])
                    self._position += 1
            elif commit_hash in planned[self._position:]:
                # A future item finished first (worker out of order): park it
                # but do not move the cursor over the gap.
                self._delivered.add(commit_hash)
                advanced = False
            elif commit_hash in planned[: self._position]:
                raise SelectionPlanError(
                    f"Commit {commit_hash} was already delivered: the cursor "
                    "cannot move backwards"
                )
            else:
                raise SelectionPlanError(
                    f"Commit {commit_hash} is not part of selection plan "
                    f"{self.plan_id}: refusing to advance the cursor"
                )

            self._auto_save()
            return advanced

    def bind(self, plan: SelectionPlan) -> None:
        """
        Bind (or re-bind) the cursor to a plan, verifying that it is the
        plan the cursor was created for and that the recorded state is
        consistent with it.
        """
        with self._lock:
            if plan.plan_id != self.plan_id:
                raise SelectionPlanError(
                    "The resume cursor belongs to a different selection plan "
                    f"(cursor: {self.plan_id[:12]}, plan: {plan.plan_id[:12]}). "
                    "Refusing to silently switch to another selection."
                )
            self._plan = plan
            self._validate_state()

    def _checksum(self) -> str:
        payload = "{plan}\n{position}\n{delivered}".format(
            plan=self.plan_id,
            position=self._position,
            delivered="\n".join(sorted(self._delivered)),
        )
        return _checksum("resume-cursor:" + payload)

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "version": _SCHEMA_VERSION,
                "plan_id": self.plan_id,
                "position": self._position,
                "delivered": sorted(self._delivered),
                "checksum": self._checksum(),
            }

    def save(self, path: Union[str, os.PathLike]) -> None:
        """
        Atomically persist the cursor to ``path``. From now on every
        :meth:`mark_delivered` is persisted to the same file, so the cursor
        stays crash-safe.
        """
        with self._lock:
            target = Path(path)
            _atomic_write_json(target, self.to_dict())
            self._path = target
            logger.info("Saved resume cursor for %s at position %d/%d",
                        self.plan_id[:12], self._position, len(self._plan))

    def _auto_save(self) -> None:
        if self._path is not None:
            _atomic_write_json(self._path, self.to_dict())

    @classmethod
    def load(
        cls,
        path: Union[str, os.PathLike],
        plan: SelectionPlan,
    ) -> "ResumeCursor":
        """
        Load a cursor from ``path`` and verify it end to end: file schema,
        tamper-evident checksum, binding to ``plan``, and state consistency.
        """
        data = _read_json(path)
        if data.get("version") != _SCHEMA_VERSION:
            raise SelectionPlanError(
                f"Unsupported resume cursor version: {data.get('version')!r}"
            )
        for required in ("plan_id", "position", "delivered", "checksum"):
            if required not in data:
                raise SelectionPlanError(
                    f"Resume cursor is missing the '{required}' field"
                )

        plan_id = data["plan_id"]
        position = data["position"]
        delivered = data["delivered"]
        if not isinstance(plan_id, str):
            raise SelectionPlanError("Resume cursor has an invalid 'plan_id'")
        if not isinstance(position, int) or isinstance(position, bool):
            raise SelectionPlanError("Resume cursor has an invalid 'position'")
        if not isinstance(delivered, list) or not all(
            isinstance(item, str) for item in delivered
        ):
            raise SelectionPlanError("Resume cursor has an invalid 'delivered' list")

        if plan_id != plan.plan_id:
            raise SelectionPlanError(
                "The resume cursor belongs to a different selection plan "
                f"(cursor: {plan_id[:12]}, plan: {plan.plan_id[:12]}). "
                "Refusing to silently switch to another selection."
            )

        cursor = cls.__new__(cls)
        cursor._lock = threading.RLock()
        cursor._path = None
        cursor._plan = plan
        cursor.plan_id = plan_id
        cursor._position = position
        cursor._delivered = set(delivered)

        stored_checksum = data["checksum"]
        if not isinstance(stored_checksum, str) or cursor._checksum() != stored_checksum:
            raise SelectionPlanError(
                "Resume cursor checksum verification failed: the file was "
                "corrupted or modified. Refusing to resume from an "
                "untrustworthy cursor."
            )

        cursor._validate_state()
        cursor._path = Path(path)
        logger.info("Loaded resume cursor for %s at position %d/%d",
                    plan_id[:12], position, len(plan))
        return cursor
