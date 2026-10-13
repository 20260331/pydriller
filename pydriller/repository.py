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
This module includes 1 class, Repository, main class of PyDriller.
"""

import logging
import math
import os
import shutil
import tempfile
import concurrent.futures
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from git import Repo
from git.exc import GitError

from pydriller.domain.commit import Commit
from pydriller.git import Git
from pydriller.selection import ResumeCursor, SelectionPlan, SelectionPlanError
from pydriller.utils.conf import Conf

logger = logging.getLogger(__name__)


class Repository:
    """
    This is the main class of PyDriller, responsible for running the study.
    """

    def __init__(self, path_to_repo: Union[str, os.PathLike, List[Union[str, os.PathLike]]],
                 single: Optional[str] = None,
                 since: Optional[datetime] = None, since_as_filter: Optional[datetime] = None, to: Optional[datetime] = None,
                 from_commit: Optional[str] = None, to_commit: Optional[str] = None,
                 from_tag: Optional[str] = None, to_tag: Optional[str] = None,
                 include_refs: bool = False,
                 include_remotes: bool = False,
                 num_workers: int = 1,
                 only_in_branch: Optional[str] = None,
                 only_modifications_with_file_types: Optional[List[str]] = None,
                 only_no_merge: bool = False,
                 only_authors: Optional[List[str]] = None,
                 only_commits: Optional[List[str]] = None,
                 only_releases: bool = False,
                 filepath: Optional[str] = None,
                 include_deleted_files: bool = False,
                 histogram_diff: bool = False,
                 skip_whitespaces: bool = False,
                 clone_repo_to: Optional[str] = None,
                 order: Optional[str] = None,
                 use_mailmap: bool = False,
                 selection_plan: Optional[Union[str, os.PathLike, SelectionPlan]] = None,
                 resume_cursor: Optional[Union[str, os.PathLike, ResumeCursor]] = None):
        """
        Init a repository. The only required parameter is
        "path_to_repo": to analyze a single repo, pass the absolute path to
        the repo; if you need to analyze more repos, pass a list of absolute
        paths.

        Furthermore, PyDriller supports local and remote repositories: if
        you pass a path to a repo, PyDriller will run the study on that
        repo; if you pass an URL, PyDriller will clone the repo in a
        temporary folder, run the study, and delete the temporary folder.

        :param path_to_repo: absolute path (or list of absolute paths) to the
            repository(ies) to analyze. Each path can be a string or any
            os.PathLike object (e.g. a pathlib.Path); remote repositories must
            be passed as a URL string.
        :param str single: hash of a single commit to analyze
        :param datetime since: starting date
        :param datetime since_as_filter: starting date (scans all commits, does not stop at first commit with date < since_as_filter)
        :param datetime to: ending date
        :param str from_commit: starting commit (only if `since` is None)
        :param str to_commit: ending commit (only if `to` is None)
        :param str from_tag: starting the analysis from specified tag (only
            if `since` and `from_commit` are None)
        :param str to_tag: ending the analysis from specified tag (only if
            `to` and `to_commit` are None)
        :param bool include_refs: whether to include refs and HEAD in commit analysis
        :param bool include_remotes: whether to include remote commits in analysis
        :param int num_workers: number of workers (i.e., threads). Please note, if num_workers > 1 the commits order is not maintained.
        :param str only_in_branch: only commits in this branch will be analyzed
        :param List[str] only_modifications_with_file_types: only
            modifications with that file types will be analyzed
        :param bool only_no_merge: if True, merges will not be analyzed
        :param List[str] only_authors: only commits of these authors will be
            analyzed (the check is done on the username, NOT the email)
        :param List[str] only_commits: only these commits will be analyzed
        :param bool only_releases: analyze only tagged commits
        :param bool histogram_diff: add the "--histogram" option when asking for the diff
        :param bool skip_whitespaces: add the "-w" option when asking for the diff
        :param str clone_repo_to: if the repo under analysis is remote, clone the repo to the specified directory
        :param str filepath: only commits that modified this file will be analyzed
        :param bool include_deleted_files: include commits modifying a deleted file (useful when analyzing a deleted `filepath`)
        :param str order: order of commits. It can be one of: 'date-order',
            'author-date-order', 'topo-order', or 'reverse'. If order=None, PyDriller returns the commits from the oldest to the newest.
        :param bool use_mailmap: Map names and emails of authors and committers to canonical values when a .mailmap file is present in
            the repository.
        :param selection_plan: fixed selection plan for the traversal. It can be a
            :class:`~pydriller.selection.SelectionPlan` object or a path to a JSON
            plan file. If the file does not exist, the ordered commits selected now
            are saved there; if it exists, the traversal resumes from that exact
            plan and never silently switches to a newer history. Only supported
            when analyzing a single repository.
        :param resume_cursor: resume cursor tracking how many planned commits were
            already delivered. It can be a :class:`~pydriller.selection.ResumeCursor`
            object or a path to a JSON cursor file. If omitted while ``selection_plan``
            is a path, the cursor is stored next to the plan as
            ``<plan>.cursor.json``.
        """
        file_modification_set = (
            None if only_modifications_with_file_types is None
            else set(only_modifications_with_file_types)
            )
        commit_set = (
            None if only_commits is None
            else set(only_commits)
            )

        options = {
            "git": None,
            "path_to_repo": path_to_repo,
            "from_commit": from_commit,
            "to_commit": to_commit,
            "from_tag": from_tag,
            "to_tag": to_tag,
            "since": since,
            "since_as_filter": since_as_filter,
            "to": to,
            "single": single,
            "include_refs": include_refs,
            "include_remotes": include_remotes,
            "num_workers": num_workers,
            "only_in_branch": only_in_branch,
            "only_modifications_with_file_types": file_modification_set,
            "only_no_merge": only_no_merge,
            "only_authors": only_authors,
            "only_commits": commit_set,
            "only_releases": only_releases,
            "skip_whitespaces": skip_whitespaces,
            "filepath": filepath,
            "include_deleted_files": include_deleted_files,
            "filepath_commits": None,
            "tagged_commits": None,
            "histogram": histogram_diff,
            "clone_repo_to": clone_repo_to,
            "order": order,
            "use_mailmap": use_mailmap,
            "selection_plan": selection_plan,
            "resume_cursor": resume_cursor,
            "active_resume_cursor": None
        }
        self._conf = Conf(options)

        # If the user provides a directory where to clone the repositories,
        # make sure we do not delete the directory after the study completes
        self._cleanup = False if clone_repo_to is not None else True

    @staticmethod
    def _is_remote(repo: str) -> bool:
        return repo.startswith(("git@", "https://", "http://", "git://"))

    def _clone_remote_repo(self, tmp_folder: str, repo: str) -> str:
        repo_folder = os.path.join(tmp_folder, self._get_repo_name_from_url(repo))
        if os.path.isdir(repo_folder):
            logger.info(f"Reusing folder {repo_folder} for {repo}")
        else:
            logger.info(f"Cloning {repo} in temporary folder {repo_folder}")
            Repo.clone_from(url=repo, to_path=repo_folder)

        return repo_folder

    def _clone_folder(self) -> str:
        if self._conf.get('clone_repo_to'):
            clone_folder = str(Path(self._conf.get('clone_repo_to')))
            if not os.path.isdir(clone_folder):
                raise Exception("Not a directory: {0}".format(clone_folder))
        else:
            # Save the temporary directory so we can clean it up later
            self._tmp_dir = tempfile.TemporaryDirectory()
            clone_folder = self._tmp_dir.name
        return clone_folder

    @contextmanager
    def _prep_repo(self, path_repo: str) -> Generator[Git, None, None]:
        local_path_repo = path_repo
        if self._is_remote(path_repo):
            local_path_repo = self._clone_remote_repo(self._clone_folder(), path_repo)
        local_path_repo = str(Path(local_path_repo).expanduser().resolve())

        # when multiple repos are given in input, this variable will serve as a reminder
        # of which one we are currently analyzing
        self._conf.set_value('path_to_repo', local_path_repo)

        self.git = Git(local_path_repo, self._conf)
        # saving the Git object for further use
        self._conf.set_value("git", self.git)

        # checking that the filters are set correctly
        self._conf.sanity_check_filters()
        yield self.git

        # cleaning, this is necessary since GitPython issues on memory leaks
        self._conf.set_value("git", None)
        self.git.clear()
        self.git = None  # type: ignore

        # delete the temporary directory if created
        if self._is_remote(path_repo) and self._cleanup is True:
            assert self._tmp_dir is not None
            try:
                self._tmp_dir.cleanup()
            except (PermissionError, OSError):
                # On Windows there might be cleanup errors.
                # Manually remove files
                shutil.rmtree(self._tmp_dir.name, ignore_errors=True)

    def traverse_commits(self) -> Generator[Commit, None, None]:
        """
        Analyze all the specified commits (all of them by default), returning
        a generator of commits.

        If a selection plan is configured, the traversal does not re-read the
        live history: it replays the exact ordered commits stored in the plan
        and advances the resume cursor only for the commits actually
        delivered. Branches moving while the traversal runs therefore cannot
        silently change what gets analyzed.
        """
        if self._conf.get('selection_plan') is not None \
                and len(self._conf.get('path_to_repos')) != 1:
            raise SelectionPlanError(
                "A selection plan is bound to a single repository history: it "
                "cannot be shared across multiple repositories"
            )
        if self._conf.get('selection_plan') is None \
                and self._conf.get('resume_cursor') is not None:
            raise Exception(
                "A resume cursor requires a selection_plan: the cursor is "
                "meaningless without the fixed selection it points to"
            )

        for path_repo in self._conf.get('path_to_repos'):
            with self._prep_repo(path_repo=path_repo) as git:
                logger.info(f'Analyzing git repository in {git.path}')

                saved_plan = self._load_existing_selection_plan(
                    self._conf.get('selection_plan')
                )

                rev: Optional[Union[str, List[str]]]
                kwargs: Dict[str, Any]
                if saved_plan is not None:
                    # Resume mode: keep the filter inputs frozen in the plan
                    # instead of recomputing them from the (possibly moved)
                    # current history.
                    self._restore_frozen_filters(saved_plan)
                    rev, kwargs = None, {}
                else:
                    # Get the commits that modified the filepath. In this case, we can not use
                    # git rev-list since it doesn't have the option --follow, necessary to follow
                    # the renames. Hence, we manually call git log instead
                    if self._conf.get('filepath') is not None:
                        self._conf.set_value(
                            'filepath_commits',
                            git.get_commits_modified_file(self._conf.get('filepath'),
                                                          self._conf.get('include_deleted_files'))
                        )

                    # Gets only the commits that are tagged
                    if self._conf.get('only_releases'):
                        self._conf.set_value('tagged_commits', git.get_tagged_commits())

                    # Build the arguments to pass to git rev-list.
                    rev, kwargs = self._conf.build_args()

                plan, cursor = self._prepare_checkpoint(git, saved_plan, rev, kwargs)

                if plan is not None and cursor is not None:
                    commits_to_analyze: Generator[Commit, None, None] = \
                        self._iter_planned_commits(git, plan, cursor)
                else:
                    commits_to_analyze = git.get_list_commits(rev, **kwargs)

                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self._conf.get("num_workers")) as executor:
                        for job in executor.map(self._iter_commits, commits_to_analyze):

                            for commit in job:
                                yield commit
                finally:
                    # The traversal of this repository finished (or the
                    # consumer stopped iterating): detach the cursor so the
                    # Repository can be reused without stale checkpoint state.
                    self._conf.set_value("active_resume_cursor", None)

    @staticmethod
    def _load_existing_selection_plan(
        selection_plan: Optional[Union[str, os.PathLike, SelectionPlan]]
    ) -> Optional[SelectionPlan]:
        """
        Return the plan to resume from, or None if a new plan has to be built.
        Paths pointing to a non-existing file mean "build and save here".
        """
        if selection_plan is None:
            return None
        if isinstance(selection_plan, SelectionPlan):
            return selection_plan
        if isinstance(selection_plan, (str, os.PathLike)):
            plan_path = Path(selection_plan)
            if plan_path.exists():
                return SelectionPlan.load(plan_path)
            return None
        raise Exception(
            "selection_plan has to be a SelectionPlan, a path, or None"
        )

    def _restore_frozen_filters(self, plan: SelectionPlan) -> None:
        if plan.filepath_commits is not None:
            self._conf.set_value('filepath_commits', set(plan.filepath_commits))
        if plan.tagged_commits is not None:
            self._conf.set_value('tagged_commits', set(plan.tagged_commits))

    def _prepare_checkpoint(
        self,
        git: Git,
        saved_plan: Optional[SelectionPlan],
        rev: Optional[Union[str, List[str]]],
        kwargs: Dict,
    ) -> Tuple[Optional[SelectionPlan], Optional[ResumeCursor]]:
        """
        Resolve the plan and the cursor to use while analyzing the current
        repository. Returns (None, None) when checkpointing is disabled.
        """
        selection_plan_arg = self._conf.get('selection_plan')
        if selection_plan_arg is None:
            return None, None

        plan_path = self._checkpoint_file_path(selection_plan_arg)

        if saved_plan is None:
            # First run: freeze the selection as it is *now*.
            selected_hashes = [commit.hash
                               for commit in git.get_list_commits(rev, **kwargs)]
            plan = SelectionPlan(
                selected_hashes,
                repo=str(git.path),
                head=self._get_head_hash(git),
                tagged_commits=self._conf.get('tagged_commits'),
                filepath_commits=self._conf.get('filepath_commits'),
            )
            if plan_path is not None:
                plan.save(plan_path)
        else:
            plan = saved_plan

        # A plan is only replayable while all of its commits still exist in
        # the object store. If some disappeared, fail loudly instead of
        # falling back to the new history.
        missing = plan.missing_commits(git.objects_exist(plan.commits))
        if missing:
            preview = ", ".join(missing[:5])
            raise SelectionPlanError(
                f"{len(missing)} of {len(plan)} planned commits do not exist "
                f"anymore in the repository (first missing: {preview}). The "
                "history changed since the selection plan was created; "
                "refusing to silently traverse the new history. Build a new "
                "plan explicitly to analyze the current history."
            )

        cursor = self._prepare_cursor(plan, plan_path)
        self._conf.set_value("active_resume_cursor", cursor)
        return plan, cursor

    def _prepare_cursor(
        self,
        plan: SelectionPlan,
        plan_path: Optional[Path],
    ) -> ResumeCursor:
        resume_cursor_arg = self._conf.get('resume_cursor')

        if isinstance(resume_cursor_arg, ResumeCursor):
            resume_cursor_arg.bind(plan)
            return resume_cursor_arg

        if resume_cursor_arg is None:
            if plan_path is None:
                # In-memory plan passed directly: the cursor cannot survive
                # the process, but the traversal still follows the plan.
                return ResumeCursor(plan)
            # By default, persist the cursor next to the plan.
            cursor_path = plan_path.with_name(plan_path.stem + ".cursor.json")
        elif isinstance(resume_cursor_arg, (str, os.PathLike)):
            cursor_path = Path(resume_cursor_arg)
        else:
            raise Exception(
                "resume_cursor has to be a ResumeCursor, a path, or None"
            )

        if cursor_path.exists():
            return ResumeCursor.load(cursor_path, plan)

        cursor = ResumeCursor(plan)
        cursor.save(cursor_path)
        return cursor

    @staticmethod
    def _checkpoint_file_path(
        selection_plan: Union[str, os.PathLike, SelectionPlan]
    ) -> Optional[Path]:
        if isinstance(selection_plan, SelectionPlan):
            return None
        return Path(selection_plan)

    @staticmethod
    def _get_head_hash(git: Git) -> Optional[str]:
        try:
            return git.repo.head.commit.hexsha
        except (ValueError, GitError, TypeError):
            # Empty repository or detached/unreadable HEAD: nothing to record.
            logger.debug("Could not read HEAD while building the selection plan")
            return None

    def _iter_planned_commits(
        self,
        git: Git,
        plan: SelectionPlan,
        cursor: ResumeCursor,
    ) -> Generator[Commit, None, None]:
        """
        Yield the planned commits in the exact order stored in the plan,
        skipping the commits the cursor already accounts for. Commit objects
        are looked up by hash, so moved branches are ignored.
        """
        for index, commit_hash in enumerate(plan.commits):
            if cursor.is_pending(index, commit_hash):
                yield git.get_commit(commit_hash)

    def _iter_commits(self, commit: Commit) -> Generator[Commit, None, None]:
        logger.info(f'Commit #{commit.hash} in {commit.committer_date} from {commit.author.name}')

        cursor = self._conf.get("active_resume_cursor")

        if self._conf.is_commit_filtered(commit):
            logger.info(f'Commit #{commit.hash} filtered')
            if cursor is not None:
                # Deterministic skip: the planned item was processed and can
                # never be delivered, so advance past it.
                cursor.mark_delivered(commit.hash)
            return

        yield commit

        if cursor is not None:
            # Acknowledge only after the consumer actually received the
            # commit, so a crash before delivery makes the item replayed.
            cursor.mark_delivered(commit.hash)

    @staticmethod
    def _split_in_chunks(full_list: List[Commit], num_workers: int) -> List[List[Commit]]:
        """
        Given the list of commits return chunks of commits based on the number of workers.

        :param List[Commit] full_list: full list of commits
        :param int num_workers: number of workers (i.e., threads)
        :return: Chunks of commits
        """
        num_chunks = math.ceil(len(full_list) / num_workers)
        chunks = []
        for i in range(0, len(full_list), num_chunks):
            chunks.append(full_list[i:i + num_chunks])

        return chunks

    @staticmethod
    def _get_repo_name_from_url(url: str) -> str:
        last_slash_index = url.rfind("/")
        len_url = len(url)

        if last_slash_index < 0 or last_slash_index >= len_url - 1:
            raise MalformedUrl(f"Badly formatted url {url}")

        last_dot_index = url.rfind(".")

        if url[last_dot_index:] == ".git":
            last_suffix_index = last_dot_index
        else:
            last_suffix_index = len_url

        return url[last_slash_index + 1:last_suffix_index]


class MalformedUrl(Exception):
    def __init__(self, message):
        super().__init__(message)
