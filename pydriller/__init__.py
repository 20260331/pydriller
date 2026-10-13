# pylint: disable=C0111
from .domain.commit import Commit, ModifiedFile, ModificationType # noqa
from .domain.line_mapping import LineMapping, DiffBlock, AlignmentType # noqa
from .repository import Repository, Git    # noqa

__version__ = "2.12"
