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
This module includes 1 class, Git, representing a repository in Git.
"""

import logging
import os
import string
from pathlib import Path
from typing import List, Dict, Optional, Set, Generator, Tuple

from git import Repo, GitCommandError
from git.objects import Commit as GitCommit, Blob, Tree

from pydriller.domain.commit import Commit, ModificationType, ModifiedFile
from pydriller.utils.conf import Conf

logger = logging.getLogger(__name__)


class Git:
    """
    Class representing a repository in Git. It contains most of the logic of
    PyDriller: obtaining the list of commits, checkout, reset, etc.
    """

    def __init__(self, path: str, conf=None):
        """
        Init the Git Repository.

        :param str path: path to the repository
        """
        self.path = Path(path).expanduser().resolve()
        self.project_name = self.path.name
        self._repo = None

        # if no configuration is passed, then creates a new "emtpy" one
        # with just "path_to_repo" inside.
        if conf is None:
            conf = Conf({
                "path_to_repo": str(self.path),
                "git": self
            })

        self._conf = conf
        self._conf.set_value("main_branch", None)  # init main_branch to None

        # Initialize repository
        self._open_repository()

        # Pin the configured notes refs to their current object version and
        # load their notes. This runs when a traversal starts, hence a notes
        # ref moving later on does not affect the commits of this traversal.
        self._prepare_notes_snapshot()

    @property
    def repo(self) -> Repo:
        """
        GitPython object Repo.

        :return: Repo
        """
        if self._repo is None:
            self._open_repository()

        assert self._repo

        return self._repo

    def clear(self):
        """
        According to GitPython's documentation, sometimes it leaks resources.
        This holds especially for Windows users. Hence, we need to clear the
        cache manually.
        """
        if self._repo:
            self.repo.git.clear_cache()

    def _open_repository(self):
        self._repo = Repo(str(self.path))
        self._repo.config_writer().set_value("blame", "markUnblamableLines", "true").release()
        if self._conf.get("main_branch") is None:
            self._discover_main_branch(self._repo)

    def _discover_main_branch(self, repo):
        try:
            self._conf.set_value("main_branch", repo.active_branch.name)
        except TypeError:
            # The current HEAD is detached. In this case, it doesn't belong to
            # any branch, hence we return an empty string
            logger.info("HEAD is a detached symbolic reference, setting main branch to empty string")
            self._conf.set_value("main_branch", '')

    def get_head(self) -> Commit:
        """
        Get the head commit.

        :return: Commit of the head commit
        """
        head_commit = self.repo.head.commit
        return Commit(head_commit, self._conf)

    def get_list_commits(self, rev='HEAD', **kwargs) -> Generator[Commit, None, None]:
        """
        Return a generator of commits of all the commits in the repo.

        :return: Generator[Commit], the generator of all the commits in the
            repo
        """
        # If not specified otherwise, analyze the repository in reversed order
        if 'reverse' not in kwargs:
            kwargs['reverse'] = True

        try:
            for commit in self.repo.iter_commits(rev=rev, **kwargs):
                yield self.get_commit_from_gitpython(commit)
        except GitCommandError as gce:
            if "fatal: bad revision 'HEAD'" in str(gce):
                logger.debug(f"Could not find commits in {self.path}")
            else:
                raise Exception(f"Error while getting commits: {gce}")

    def get_commit(self, commit_id: str) -> Commit:
        """
        Get the specified commit.

        :param str commit_id: hash of the commit to analyze
        :return: Commit
        """
        gp_commit = self.repo.commit(commit_id)
        return Commit(gp_commit, self._conf)

    def get_commit_from_gitpython(self, commit: GitCommit) -> Commit:
        """
        Build a PyDriller commit object from a GitPython commit object.
        This is internal of PyDriller, I don't think users generally will need
        it.

        :param GitCommit commit: GitPython commit
        :return: Commit commit: PyDriller commit
        """
        return Commit(commit, self._conf)

    def checkout(self, _hash: str) -> None:
        """
        Checkout the repo at the speficied commit.
        BE CAREFUL: this will change the state of the repo, hence it should
        *not* be used with more than 1 thread.

        :param _hash: commit hash to checkout
        """
        self.repo.git.checkout('-f', _hash)

    def files(self) -> List[str]:
        """
        Obtain the list of the files (excluding .git directory).

        :return: List[str], the list of the files
        """
        _all = []
        for path, _, files in os.walk(str(self.path)):
            if '.git' in path:
                continue
            for name in files:
                _all.append(os.path.join(path, name))
        return _all

    def reset(self) -> None:
        """
        Reset the state of the repo, checking out the main branch and
        discarding
        local changes (-f option).

        """
        self.repo.git.checkout('-f', self._conf.get("main_branch"))

    def total_commits(self) -> int:
        """
        Calculate total number of commits.

        :return: the total number of commits
        """
        return len(list(self.get_list_commits()))

    def get_commit_from_tag(self, tag: str) -> Commit:
        """
        Obtain the tagged commit.

        :param str tag: the tag
        :return: Commit commit: the commit the tag referred to
        """
        try:
            selected_tag = self.repo.tags[tag]
            return self.get_commit(selected_tag.commit.hexsha)
        except (IndexError, AttributeError):
            logger.debug(f"Tag {tag} not found")
            raise

    def get_tagged_commits(self):
        """
        Obtain the hash of all the tagged commits.

        :return: list of tagged commits (can be empty if there are no tags)
        """
        tags = []
        for tag in self.repo.tags:
            if tag.commit:
                tags.append(tag.commit.hexsha)
        return tags

    def get_commits_last_modified_lines(self, commit: Commit,
                                        modification: Optional[ModifiedFile] = None,
                                        hashes_to_ignore_path: Optional[str] = None) \
            -> Dict[str, Set[str]]:
        """
        Given the Commit object, returns the set of commits that last
        "touched" the lines that are modified in the files included in the
        commit. It applies SZZ.

        The algorithm works as follow: (for every file in the commit)

        1- obtain the diff

        2- obtain the list of deleted lines

        3- blame the file and obtain the commits were those lines were added

        Can also be passed as parameter a single Modification, in this case
        only this file will be analyzed.

        :param Commit commit: the commit to analyze
        :param Modification modification: single modification to analyze
        :param str hashes_to_ignore_path: path to a file containing hashes of
               commits to ignore.
        :return: Dict commits: a dictionary having as keys the files of the commit,
                 and as values the commits that last touched those files.
        """
        if modification is not None:
            modifications = [modification]
        else:
            modifications = commit.modified_files

        return self._calculate_last_commits(commit, modifications,
                                            hashes_to_ignore_path)

    def diff(self, from_commit_id: str, to_commit_id: str) -> List[ModifiedFile]:
        from_commit = self.repo.commit(from_commit_id)
        to_commit = self.repo.commit(to_commit_id)
        diff_index = from_commit.diff(
            other=to_commit,
            paths=None,
            create_patch=True)

        modified_files_list = [ModifiedFile(diff=diff) for diff in diff_index]
        return modified_files_list

    def _calculate_last_commits(self, commit: Commit,
                                modifications: List[ModifiedFile],
                                hashes_to_ignore_path: Optional[str] = None) \
            -> Dict[str, Set[str]]:

        commits: Dict[str, Set[str]] = {}

        for mod in modifications:
            path = mod.new_path
            if mod.change_type == ModificationType.RENAME or mod.change_type == ModificationType.DELETE:
                path = mod.old_path

            deleted_lines = mod.diff_parsed['deleted']

            assert path is not None, "We could not find the path to the file"

            try:
                blame = self._get_blame(commit.hash, path, hashes_to_ignore_path)
                for num_line, line in deleted_lines:
                    if not self._useless_line(line.strip()):
                        buggy_commit = blame[num_line - 1].split(' ')[0].replace('^', '')

                        # Skip unblamable lines.
                        if buggy_commit.startswith("*"):
                            continue

                        if mod.change_type == ModificationType.RENAME:
                            path = mod.new_path

                        assert path is not None, "We could not find the path to the file"
                        commits.setdefault(path, set()).add(self.get_commit(buggy_commit).hash)
            except GitCommandError:
                logger.debug(f"Could not found file {mod.filename} in commit {commit.hash}. Probably a double rename!")

        return commits

    def _get_blame(self, commit_hash: str, path: str, hashes_to_ignore_path: Optional[str] = None):
        args = ['-w', commit_hash + '^']
        if hashes_to_ignore_path is not None:
            if self.repo.git.version_info >= (2, 23):
                args += ["--ignore-revs-file", hashes_to_ignore_path]
            else:
                logger.info("'--ignore-revs-file' is only available from git v2.23")
        return self.repo.git.blame(*args, '--', path).split('\n')

    @staticmethod
    def _useless_line(line: str):
        # this covers comments in Java and Python, as well as empty lines.
        # More have to be added!
        return not line or \
               line.startswith('//') or \
               line.startswith('#') or \
               line.startswith("/*") or \
               line.startswith("'''") or \
               line.startswith('"""') or \
               line.startswith("*")

    def get_commits_modified_file(self, filepath: str, include_deleted_files=False) -> List[str]:
        """
        Given a filepath, returns all the commits that modified this file
        (following renames).

        :param str filepath: path to the file
        :param bool include_deleted_files: if True, include commits that modifies a deleted file
        :return: the list of commits' hash
        """
        path = str(Path(filepath))

        commits = []
        try:
            if include_deleted_files:
                commits = self.repo.git.log("--follow", "--format=%H", "--", path).split('\n')
            else:
                commits = self.repo.git.log("--follow", "--format=%H", path).split('\n')
        except GitCommandError:
            logger.debug(f"Could not find information of file {path}")

        return commits

    def _prepare_notes_snapshot(self) -> None:
        """
        Resolve each configured notes ref, pin it to its current object SHA
        and load all the notes reachable from that object.

        The snapshot is stored in the configuration as a dictionary mapping
        each notes ref to a dictionary {annotated object SHA: note text}.
        Since notes are content-addressed and the ref is resolved once,
        moving the ref after this snapshot is built cannot change the notes
        seen by the commits of the ongoing traversal.
        """
        notes_refs = self._conf.get("notes_refs") or []
        snapshot: Dict[str, Dict[str, str]] = {}

        for notes_ref in notes_refs:
            resolved_ref = self._resolve_notes_ref(notes_ref)
            if resolved_ref is None:
                logger.warning(f"Notes ref {notes_ref} not found in {self.path}, "
                               "no notes will be loaded from it")
                continue

            pinned_sha = self.repo.git.rev_parse(resolved_ref)
            logger.debug(f"Pinning notes ref {resolved_ref} at {pinned_sha}")
            snapshot[resolved_ref] = self._read_notes_at(pinned_sha)

        self._conf.set_value("notes_snapshot", snapshot)

    def _resolve_notes_ref(self, notes_ref: str) -> Optional[str]:
        """
        Resolve a notes ref provided by the user to its full ref name.

        The ref is accepted as-is if it exists, otherwise it is looked up
        under "refs/notes/" (e.g. "commits" becomes "refs/notes/commits"),
        mirroring the behavior of "git notes --ref".

        :param str notes_ref: notes ref as provided by the user
        :return: the full ref name, or None if the ref does not exist
        """
        try:
            self.repo.git.rev_parse('--verify', notes_ref)
            return notes_ref
        except GitCommandError:
            pass

        short_ref = notes_ref
        prefix = "refs/notes/"
        if short_ref.startswith(prefix):
            short_ref = short_ref[len(prefix):]
        candidate = prefix + short_ref

        try:
            self.repo.git.rev_parse('--verify', candidate)
            return candidate
        except GitCommandError:
            return None

    def _read_notes_at(self, notes_commit_sha: str) -> Dict[str, str]:
        """
        Load all the notes of the notes commit object identified by
        `notes_commit_sha`, returning a dictionary mapping the SHA of each
        annotated object to the text of its note.

        Notes can be stored at the root of the notes tree or in fan-out
        subdirectories (e.g. "ab/cdef..."); the components of the path are
        joined to reconstruct the annotated object SHA.

        :param str notes_commit_sha: SHA of the (pinned) notes commit
        :return: dictionary {annotated object SHA: note text}
        """
        notes: Dict[str, str] = {}
        notes_commit = self.repo.commit(notes_commit_sha)

        for annotated_sha, blob in self._iter_note_blobs(notes_commit.tree):
            notes[annotated_sha] = blob.data_stream.read().decode("utf-8", "ignore")

        return notes

    @classmethod
    def _iter_note_blobs(cls, tree: Tree, prefix: str = "") -> Generator[Tuple[str, Blob], None, None]:
        """
        Recursively iterate over the note blobs of a notes tree, pairing
        each blob with the SHA of the object it annotates.
        """
        for item in tree:
            entry_path = prefix + item.name
            if isinstance(item, Tree):
                yield from cls._iter_note_blobs(item, entry_path)
            elif isinstance(item, Blob):
                annotated_sha = entry_path.replace("/", "")
                if len(annotated_sha) == 40 and all(c in string.hexdigits for c in annotated_sha):
                    yield annotated_sha.lower(), item

    def __del__(self):
        self.clear()
