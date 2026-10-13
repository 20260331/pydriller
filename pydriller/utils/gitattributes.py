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
Read and evaluate gitattributes rules from the tree of a *specific* commit.

Classifying files (e.g. as generated or vendored) while mining historical
commits requires the rules that were in effect *at that point in history*.
Reading the ``.gitattributes`` file from the current working tree would
instead apply today's rules to the past. The classes in this module read the
``.gitattributes`` blobs stored in the commit trees through GitPython, hence
they never touch (and are not affected by) the working tree, the index, or
``$GIT_DIR/info/attributes``.

Attribute states mirror the output of ``git check-attr``:

* ``True``: the attribute is set (``foo``)
* ``False``: the attribute is unset (``-foo``)
* a ``str``: the attribute is set to that value (``foo=bar``)
* unspecified attributes (``!foo``) are simply omitted from the result
"""

import logging
import os
import re
import shlex
from typing import Any, Dict, List, Optional, Tuple, Union

from git import Repo
from git.objects import Commit as GitCommit

logger = logging.getLogger(__name__)

#: State of an attribute that is set (``foo``).
SET = True
#: State of an attribute that is unset (``-foo``).
UNSET = False
#: State of an attribute that is explicitly unspecified (``!foo``).
UNSPECIFIED: Optional[bool] = None

AttributeState = Optional[Union[bool, str]]


class _Rule:
    """
    A single pattern together with the attribute states it assigns to the
    paths it matches.
    """

    __slots__ = ("regex", "directory_only", "attributes")

    def __init__(self, regex: "re.Pattern[str]", directory_only: bool,
                 attributes: List[Tuple[str, AttributeState]]):
        self.regex = regex
        self.directory_only = directory_only
        self.attributes = attributes

    def matches(self, relative_path: str) -> bool:
        """
        :param relative_path: path of the candidate file relative to the
            directory containing the .gitattributes file this rule comes from
        :return: True if the pattern matches the path
        """
        return self.regex.match(relative_path) is not None


def _translate_pattern(pattern: str) -> str:
    """
    Translate a git-style glob (relative to the .gitattributes directory)
    into the body of a regular expression.
    """
    out: List[str] = []
    i = 0
    length = len(pattern)
    while i < length:
        char = pattern[i]
        if char == '*':
            if i + 1 < length and pattern[i + 1] == '*':
                i += 2
                # "**/" matches zero or more leading directories
                if i < length and pattern[i] == '/':
                    i += 1
                    out.append(r'(?:.*/)?')
                else:
                    out.append(r'.*')
            else:
                i += 1
                out.append(r'[^/]*')
        elif char == '?':
            i += 1
            out.append(r'[^/]')
        elif char == '[':
            j = i + 1
            if j < length and pattern[j] == '!':
                j += 1
            if j < length and pattern[j] == ']':
                j += 1
            while j < length and pattern[j] != ']':
                j += 1
            if j >= length:
                # No closing bracket: "[" is a literal character
                out.append(r'\[')
                i += 1
            else:
                body = pattern[i + 1:j]
                if body.startswith('!'):
                    body = '^' + body[1:]
                out.append('[' + body + ']')
                i = j + 1
        else:
            out.append(re.escape(char))
            i += 1
    return ''.join(out)


def _compile_rule(pattern: str,
                  attributes: List[Tuple[str, AttributeState]]) -> Optional[_Rule]:
    """
    Compile a .gitattributes pattern. Pattern anchoring follows the same
    rules documented in gitattributes(5):

    * a trailing "/" limits the match to directories (never matches the
      files queried by PyDriller, hence no rule is returned)
    * a leading "/" anchors the pattern to the .gitattributes directory
    * any other "/" in the pattern anchors it to that directory as well
    * a pattern without "/" matches the basename at any depth below the
      .gitattributes directory
    """
    directory_only = False
    if pattern.endswith('/'):
        directory_only = True
        pattern = pattern[:-1]
    if not pattern:
        return None

    if pattern.startswith('/'):
        anchored = True
        pattern = pattern[1:]
    elif '/' in pattern:
        anchored = True
    else:
        anchored = False

    body = _translate_pattern(pattern)
    if anchored:
        regex = re.compile('^' + body + '$')
    else:
        regex = re.compile(r'^(?:.*/)?' + body + '$')

    return _Rule(regex, directory_only, attributes)


def _parse_attribute(token: str) -> Tuple[str, AttributeState]:
    if len(token) > 1 and token[0] == '-':
        return token[1:], UNSET
    if len(token) > 1 and token[0] == '!':
        return token[1:], UNSPECIFIED
    if '=' in token:
        name, value = token.split('=', 1)
        return name, value
    return token, SET


def _parse_attributes_file(content: bytes) -> List[_Rule]:
    """
    Parse the content of a .gitattributes blob into a list of rules,
    keeping their original order (later rules override earlier ones).
    """
    rules: List[_Rule] = []
    text = content.decode("utf-8", "ignore")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue

        lexer = shlex.shlex(stripped, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ''
        lexer.quotes = '"'
        try:
            tokens = list(lexer)
        except ValueError:
            # E.g. a line with an unmatched quote: ignore it rather than
            # failing the whole historical analysis
            logger.debug("Could not parse gitattributes line: %s", stripped)
            continue

        if len(tokens) < 2:
            # A pattern with no attribute state has no effect
            continue

        pattern = tokens[0]
        states = [_parse_attribute(token) for token in tokens[1:]]
        rule = _compile_rule(pattern, states)
        if rule is not None and not rule.directory_only:
            rules.append(rule)

    return rules


class GitAttributes:
    """
    Evaluate gitattributes rules as they were defined in the tree of any
    given commit. Rules are read from the commit objects through GitPython
    (never from the working tree), so the returned values reflect the
    repository history rather than its current state.

    Parsed .gitattributes files are cached by blob hash: a file that is
    unchanged across commits is parsed only once.
    """

    def __init__(self, repo: Repo):
        self._repo = repo
        self._rules_cache: Dict[str, List[_Rule]] = {}

    def get_attributes(self, commit: Union[str, GitCommit],
                       path: Union[str, os.PathLike]) -> Dict[str, Any]:
        """
        Return the attributes assigned to ``path`` by the ``.gitattributes``
        files found in the tree of ``commit``.

        ``$GIT_DIR/info/attributes`` and global/system files are not
        consulted: only the rules versioned in the commit tree are used.
        Attributes left unspecified are not present in the returned
        dictionary.

        :param commit: hash of the commit or GitPython Commit object
        :param path: repository-relative path (e.g. the ``new_path`` of a
            ModifiedFile)
        :return: a dictionary mapping attribute names to ``True`` (set),
            ``False`` (unset), or a string (set to a value)
        """
        if isinstance(commit, str):
            commit = self._repo.commit(commit)

        normalized = str(path).replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        while "//" in normalized:
            normalized = normalized.replace("//", "/")
        normalized = normalized.strip("/")
        if not normalized:
            return {}

        tree = commit.tree

        # .gitattributes files are consulted from the root of the tree down
        # to the directory containing the path; deeper files (and later
        # lines) override shallower ones.
        directories = [""]
        current = ""
        for component in normalized.split("/")[:-1]:
            current = f"{current}/{component}" if current else component
            directories.append(current)

        resolved: Dict[str, AttributeState] = {}
        for directory in directories:
            relative_path = (
                normalized[len(directory) + 1:] if directory else normalized
            )
            for rule in self._get_rules(tree, directory):
                if rule.matches(relative_path):
                    for name, state in rule.attributes:
                        resolved[name] = state

        return {
            name: state
            for name, state in resolved.items()
            if state is not UNSPECIFIED
        }

    def _get_rules(self, tree: Any, directory: str) -> List[_Rule]:
        attributes_path = (
            f"{directory}/.gitattributes" if directory else ".gitattributes"
        )
        try:
            blob = tree[attributes_path]
        except KeyError:
            return []

        blob_hash = blob.hexsha
        rules = self._rules_cache.get(blob_hash)
        if rules is None:
            rules = _parse_attributes_file(blob.data_stream.read())
            self._rules_cache[blob_hash] = rules
        return rules
