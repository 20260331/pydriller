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
Tests for the raw signature envelope support (gpgsig / mergetag headers,
byte-exact signed payloads and the pluggable offline verifier).
"""
import base64
import datetime as dt
import struct
import subprocess

import pytest

from pydriller.domain.signature import (
    ArmorKind,
    MergeTagEnvelope,
    OfflineSignatureVerifier,
    OpenPGPSignatureInfo,
    SignatureEnvelope,
    SignatureScope,
    SignatureVerifier,
    SSHSignatureInfo,
    VerificationResult,
    VerificationStatus,
    parse_commit_envelopes,
    parse_tag_envelopes,
)
from pydriller.git import Git

TREE = "1111111111111111111111111111111111111111"
PARENT = "2222222222222222222222222222222222222222"
ACTOR_LINE = "A U Thor <author@example.com> 1700000000 +0000"

BASE_HEADERS = (
    "tree {tree}\n"
    "parent {parent}\n"
    "author {actor}\n"
    "committer {actor}\n"
).format(tree=TREE, parent=PARENT, actor=ACTOR_LINE).encode()


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def fold_header(name: bytes, value: bytes) -> bytes:
    """Fold a multi-line header value exactly like Git's add_extra_header()
    does: a space prefixes every line, line terminators are kept and the
    block is completed with a newline."""
    lines = value.splitlines(keepends=True)
    block = name + b" " + lines[0]
    for line in lines[1:]:
        block += b" " + line
    if not block.endswith(b"\n"):
        block += b"\n"
    return block


def openpgp_packet(
        sig_type: int = 0x00,
        pk_algo: int = 22,
        hash_algo: int = 10,
        hashed: bytes = b"",
        unhashed: bytes = b"",
        trailer: bytes = b"\x12\x34" + b"\x00" * 10) -> bytes:
    body = bytes([4, sig_type, pk_algo, hash_algo])
    body += struct.pack(">H", len(hashed)) + hashed
    body += struct.pack(">H", len(unhashed)) + unhashed
    body += trailer
    # New-format packet header, tag 2 (signature), 4-octet length.
    return b"\xc2\xff" + struct.pack(">I", len(body)) + body


def subpacket(sub_type: int, data: bytes, critical: bool = False) -> bytes:
    body = bytes([sub_type | (0x80 if critical else 0)]) + data
    length = len(body)
    if length < 192:
        return bytes([length]) + body
    first = 192 + (length - 192) // 256
    return bytes([first, (length - 192) % 256]) + body


def armor_openpgp(packet: bytes, headers: bytes = b"") -> bytes:
    b64 = base64.b64encode(packet).decode()
    text = "-----BEGIN PGP SIGNATURE-----\n"
    if headers:
        text += headers.decode() + "\n"
    text += "\n" + b64 + "\n"
    text += "\n=AAAA\n"  # Radix-64 CRC24 checksum line, must be ignored
    text += "-----END PGP SIGNATURE-----"
    return text.encode()


def armor_ssh(blob: bytes) -> bytes:
    return (b"-----BEGIN SSH SIGNATURE-----\n\n"
            + base64.b64encode(blob)
            + b"\n-----END SSH SIGNATURE-----")


def ssh_string(value: bytes) -> bytes:
    return struct.pack(">I", len(value)) + value


def ssh_signature_blob() -> bytes:
    public_key = ssh_string(b"ssh-ed25519") + b"\xab" * 32
    signature = ssh_string(b"ssh-ed25512")  # inner signature algorithm
    return (
        ssh_string(b"SSHSIG")
        + struct.pack(">I", 1)
        + ssh_string(public_key)
        + ssh_string(b"git")
        + ssh_string(b"")
        + ssh_string(b"sha512")
        + ssh_string(signature)
    )


# ---------------------------------------------------------------------------
# Header folding / unsigned commits
# ---------------------------------------------------------------------------

def test_unsigned_commit_has_no_envelopes():
    raw = BASE_HEADERS + b"\ninitial commit\n"
    signatures, merge_tags = parse_commit_envelopes(raw)

    assert signatures == ()
    assert merge_tags == ()


def test_header_without_space_is_tolerated():
    raw = b"tree x\nbareheader\ngpgsig \xc3\xa8\xc3\xa9\n\nmsg\n"
    signatures, _ = parse_commit_envelopes(raw)
    assert signatures[0].signature_bytes == "èé\n".encode()
    assert signatures[0].armor == ArmorKind.UNKNOWN


# ---------------------------------------------------------------------------
# gpgsig: unfolding and byte-exact signed payload
# ---------------------------------------------------------------------------

def test_gpgsig_envelope_unfolds_armor_byte_for_byte():
    armor = armor_openpgp(openpgp_packet())
    raw = BASE_HEADERS + fold_header(b"gpgsig", armor) + b"\nsigned commit\n"

    signatures, _ = parse_commit_envelopes(raw)
    assert len(signatures) == 1

    envelope = signatures[0]
    assert envelope.scope == SignatureScope.COMMIT
    assert envelope.header == "gpgsig"
    assert envelope.armor == ArmorKind.OPENPGP
    # Git terminates the folded block with a newline, which is restored.
    assert envelope.signature_bytes == armor + b"\n"


def test_signed_payload_is_raw_body_without_signature_block():
    armor = armor_openpgp(openpgp_packet())
    signature_block = fold_header(b"gpgsig", armor)
    raw = BASE_HEADERS + signature_block + b"\nsigned commit\n"

    signatures, _ = parse_commit_envelopes(raw)
    envelope = signatures[0]

    # Git removes exactly the folded header byte range before verifying.
    assert envelope.signed_payload == BASE_HEADERS + b"\nsigned commit\n"
    assert signature_block not in envelope.signed_payload
    # The payload is itself a well-formed, unsigned commit body.
    payload_signatures, payload_merge_tags = parse_commit_envelopes(
        envelope.signed_payload
    )
    assert payload_signatures == ()
    assert payload_merge_tags == ()


def test_signed_payload_when_signature_is_not_last_header():
    armor = armor_openpgp(openpgp_packet())
    signature_block = fold_header(b"gpgsig", armor)
    encoding_block = b"encoding ISO-8859-1\n"
    raw = BASE_HEADERS + signature_block + encoding_block + b"\nmsg\n"

    signatures, _ = parse_commit_envelopes(raw)
    assert signatures[0].signed_payload == BASE_HEADERS + encoding_block + b"\nmsg\n"


def test_gpgsig_sha256_header_is_supported():
    armor = armor_openpgp(openpgp_packet())
    raw = BASE_HEADERS + fold_header(b"gpgsig-sha256", armor) + b"\nmsg\n"

    signatures, _ = parse_commit_envelopes(raw)
    assert signatures[0].header == "gpgsig-sha256"
    assert signatures[0].signed_payload == BASE_HEADERS + b"\nmsg\n"


# ---------------------------------------------------------------------------
# OpenPGP packet metadata
# ---------------------------------------------------------------------------

def test_openpgp_metadata_is_extracted_offline():
    hashed = (
        subpacket(0x02, struct.pack(">I", 1700000000))
        + subpacket(0x21, b"\x04" + b"\xab" * 20)  # v4 issuer fingerprint
    )
    unhashed = subpacket(0x10, b"\xde" * 8)  # issuer key id
    packet = openpgp_packet(hashed=hashed, unhashed=unhashed)

    signatures, _ = parse_commit_envelopes(
        BASE_HEADERS + fold_header(b"gpgsig", armor_openpgp(packet)) + b"\nm\n"
    )
    info = signatures[0].metadata

    assert isinstance(info, OpenPGPSignatureInfo)
    assert info.version == 4
    assert info.signature_type == 0x00
    assert "Binary document" in info.signature_type_name
    assert info.public_key_algorithm == 22
    assert info.public_key_algorithm_name == "EdDSA"
    assert info.hash_algorithm == 10
    assert info.hash_algorithm_name == "SHA-512"
    assert info.creation_time == dt.datetime(2023, 11, 14, 22, 13, 20,
                                             tzinfo=dt.timezone.utc)
    assert info.issuer_key_id == "DE" * 8
    assert info.issuer_fingerprint == "AB" * 20


def test_openpgp_hashed_subpackets_override_unhashed_ones():
    # A forged issuer in the unhashed area must not shadow the hashed one.
    hashed = subpacket(0x10, b"\x01" * 8)
    unhashed = subpacket(0x10, b"\x02" * 8)
    packet = openpgp_packet(hashed=hashed, unhashed=unhashed)

    signatures, _ = parse_commit_envelopes(
        BASE_HEADERS + fold_header(b"gpgsig", armor_openpgp(packet)) + b"\nm\n"
    )
    assert signatures[0].metadata.issuer_key_id == "01" * 8


def test_armor_headers_and_crc_line_are_ignored():
    packet = openpgp_packet()
    armor = armor_openpgp(packet, headers=b"Version: GnuPG v2")

    signatures, _ = parse_commit_envelopes(
        BASE_HEADERS + fold_header(b"gpgsig", armor) + b"\nm\n"
    )
    assert signatures[0].metadata is not None
    assert signatures[0].parse_error is None


def test_malformed_openpgp_packet_is_reported_not_raised():
    # New-format tag-2 packet claiming 30 bytes but containing only 3;
    # the version-4 parser must flag it as malformed.
    broken = b"\xc2\x1e" + b"\x04\x00\x16"
    armor = (b"-----BEGIN PGP SIGNATURE-----\n\n"
             + base64.b64encode(broken)
             + b"\n-----END PGP SIGNATURE-----")

    signatures, _ = parse_commit_envelopes(
        BASE_HEADERS + fold_header(b"gpgsig", armor) + b"\nm\n"
    )
    envelope = signatures[0]
    assert envelope.metadata is None
    assert envelope.parse_error is not None
    assert "malformed OpenPGP" in envelope.parse_error
    assert envelope.is_structurally_valid is False


def test_unrecognized_armor_is_malformed():
    armor = b"-----BEGIN MAGIC-----\nZm9v\n-----END MAGIC-----"
    raw = BASE_HEADERS + fold_header(b"gpgsig", armor) + b"\nm\n"

    signatures, _ = parse_commit_envelopes(raw)
    assert signatures[0].armor == ArmorKind.UNKNOWN
    assert signatures[0].parse_error == "unrecognized signature armor"


# ---------------------------------------------------------------------------
# SSH signatures
# ---------------------------------------------------------------------------

def test_ssh_signature_metadata_is_extracted_offline():
    armor = armor_ssh(ssh_signature_blob())
    raw = BASE_HEADERS + fold_header(b"gpgsig", armor) + b"\nm\n"

    signatures, _ = parse_commit_envelopes(raw)
    envelope = signatures[0]
    assert envelope.armor == ArmorKind.SSH

    info = envelope.metadata
    assert isinstance(info, SSHSignatureInfo)
    assert info.public_key_type == "ssh-ed25519"
    assert info.namespace == "git"
    assert info.hash_algorithm == "sha512"
    assert info.signature_algorithm == "ssh-ed25512"


def test_malformed_ssh_signature_is_reported_not_raised():
    armor = armor_ssh(ssh_string(b"NOT-SSHSIG"))
    raw = BASE_HEADERS + fold_header(b"gpgsig", armor) + b"\nm\n"

    signatures, _ = parse_commit_envelopes(raw)
    envelope = signatures[0]
    assert envelope.armor == ArmorKind.SSH
    assert envelope.metadata is None
    assert "malformed SSH" in envelope.parse_error


# ---------------------------------------------------------------------------
# mergetag headers
# ---------------------------------------------------------------------------

def test_mergetag_unfolds_to_complete_tag_body():
    tag_body = (
        "object {parent}\n"
        "type commit\n"
        "tag v1.0\n"
        "tagger {actor}\n"
        "\n"
        "Release 1.0\n"
    ).format(parent=PARENT, actor=ACTOR_LINE).encode()

    raw = BASE_HEADERS + fold_header(b"mergetag", tag_body) + b"\nmerge\n"
    _, merge_tags = parse_commit_envelopes(raw)

    assert len(merge_tags) == 1
    merge_tag = merge_tags[0]
    assert isinstance(merge_tag, MergeTagEnvelope)
    assert merge_tag.target == PARENT
    assert merge_tag.object_type == "commit"
    assert merge_tag.tag_name == "v1.0"
    assert merge_tag.tagger == ACTOR_LINE
    assert merge_tag.message == b"Release 1.0\n"
    # Unfolding is the identity function on the embedded object bytes.
    assert merge_tag.tag_bytes == tag_body
    assert merge_tag.signatures == ()


def test_signed_mergetag_preserves_nested_signed_payload():
    armor = armor_openpgp(openpgp_packet())
    tag_headers = (
        "object {parent}\n"
        "type commit\n"
        "tag v1.0\n"
        "tagger {actor}\n"
    ).format(parent=PARENT, actor=ACTOR_LINE).encode()
    tag_body = tag_headers + fold_header(b"gpgsig", armor) + b"\nRelease 1.0\n"

    raw = BASE_HEADERS + fold_header(b"mergetag", tag_body) + b"\nmerge\n"
    _, merge_tags = parse_commit_envelopes(raw)

    merge_tag = merge_tags[0]
    assert merge_tag.tag_bytes == tag_body
    assert len(merge_tag.signatures) == 1
    nested = merge_tag.signatures[0]
    assert nested.scope == SignatureScope.MERGE_TAG
    assert nested.signature_bytes == armor + b"\n"
    assert nested.signed_payload == tag_headers + b"\nRelease 1.0\n"


def test_multiple_mergetags_keep_header_order():
    tag_tpl = (
        "object {parent}\n"
        "type commit\n"
        "tag {name}\n"
        "tagger {actor}\n\n{name}\n"
    )
    blocks = b"".join(
        fold_header(
            b"mergetag",
            tag_tpl.format(parent=PARENT, name=name, actor=ACTOR_LINE).encode(),
        )
        for name in ("v1", "v2")
    )
    raw = BASE_HEADERS + blocks + b"\nmerge\n"

    _, merge_tags = parse_commit_envelopes(raw)
    assert [mt.tag_name for mt in merge_tags] == ["v1", "v2"]


def test_parse_tag_envelopes_on_standalone_tag():
    armor = armor_openpgp(openpgp_packet())
    raw = (
        b"object " + PARENT.encode() + b"\n"
        b"type commit\n"
        b"tag v1\n"
        b"tagger " + ACTOR_LINE.encode() + b"\n"
        + fold_header(b"gpgsig", armor)
        + b"\nmsg\n"
    )
    signatures, message = parse_tag_envelopes(raw)
    assert len(signatures) == 1
    assert signatures[0].scope == SignatureScope.MERGE_TAG
    assert message == b"msg\n"


# ---------------------------------------------------------------------------
# Verifier interface
# ---------------------------------------------------------------------------

def test_offline_verifier_marks_valid_packets_unverified():
    hashed = (
        subpacket(0x02, struct.pack(">I", 1700000000))
        + subpacket(0x21, b"\x04" + b"\xab" * 20)
    )
    unhashed = subpacket(0x10, b"\xde" * 8)
    packet = openpgp_packet(hashed=hashed, unhashed=unhashed)
    signatures, _ = parse_commit_envelopes(
        BASE_HEADERS + fold_header(b"gpgsig", armor_openpgp(packet)) + b"\nm\n"
    )

    result = OfflineSignatureVerifier().verify(signatures[0])

    assert isinstance(result, VerificationResult)
    assert result.status == VerificationStatus.UNVERIFIED
    assert result.verified is None
    assert result.key_id == "DE" * 8
    assert result.fingerprint == "AB" * 20
    assert result.creation_time is not None
    assert "offline" in result.reason


def test_offline_verifier_marks_malformed_packets():
    armor = b"-----BEGIN MAGIC-----\nZm9v\n-----END MAGIC-----"
    signatures, _ = parse_commit_envelopes(
        BASE_HEADERS + fold_header(b"gpgsig", armor) + b"\nm\n"
    )

    result = OfflineSignatureVerifier().verify(signatures[0])
    assert result.status == VerificationStatus.MALFORMED
    assert result.verified is False


class _RecordingVerifier(SignatureVerifier):
    def __init__(self):
        self.seen = []

    def verify(self, envelope: SignatureEnvelope) -> VerificationResult:
        self.seen.append(envelope)
        return VerificationResult(
            envelope=envelope,
            status=VerificationStatus.VERIFIED,
            verified=True,
            reason="plugged-in test verifier",
        )


def test_custom_verifier_can_be_plugged_in():
    armor = armor_openpgp(openpgp_packet())
    signatures, _ = parse_commit_envelopes(
        BASE_HEADERS + fold_header(b"gpgsig", armor) + b"\nm\n"
    )

    verifier = _RecordingVerifier()
    result = verifier.verify(signatures[0])

    assert result.status == VerificationStatus.VERIFIED
    assert result.verified is True
    assert verifier.seen == [signatures[0]]


# ---------------------------------------------------------------------------
# Integration on a real Git repository (no gpg involved: signed objects are
# injected with "git hash-object --literally")
# ---------------------------------------------------------------------------

def _git(cwd, *args, _input=None):
    return subprocess.run(
        ["git", "-C", str(cwd)] + list(args),
        check=True, capture_output=True, input=_input
    )


@pytest.fixture
def signed_repo(tmp_path):
    _git(tmp_path, "-c", "user.name=Tester", "-c", "user.email=t@example.com",
         "init", "-q")
    (tmp_path / "file.txt").write_text("hello\n")
    _git(tmp_path, "-c", "user.name=Tester", "-c", "user.email=t@example.com",
         "add", "file.txt")
    _git(tmp_path, "-c", "user.name=Tester", "-c", "user.email=t@example.com",
         "commit", "-qm", "base")
    tree = _git(tmp_path, "rev-parse", "HEAD^{tree}").stdout.decode().strip()
    parent = _git(tmp_path, "rev-parse", "HEAD").stdout.decode().strip()
    return tmp_path, tree, parent


def _write_commit(repo_path, body: bytes) -> str:
    result = _git(repo_path, "hash-object", "-t", "commit", "-w",
                  "--literally", "--stdin", _input=body)
    return result.stdout.decode().strip()


def test_commit_exposes_raw_object_and_signature_envelopes(signed_repo):
    path, tree, parent = signed_repo
    armor = armor_openpgp(openpgp_packet(
        hashed=subpacket(0x10, b"\x42" * 8)))
    body = (
        ("tree %s\nparent %s\nauthor %s\ncommitter %s\n"
         % (tree, parent, ACTOR_LINE, ACTOR_LINE)).encode()
        + fold_header(b"gpgsig", armor)
        + b"\nsigned commit\n"
    )
    sha = _write_commit(path, body)

    gr = Git(str(path))
    try:
        commit = gr.get_commit(sha)

        # Raw bytes must match what git itself serves.
        cat_file = _git(path, "cat-file", "commit", sha).stdout
        assert commit.raw_commit == cat_file == body

        assert len(commit.signatures) == 1
        envelope = commit.signatures[0]
        assert envelope.armor == ArmorKind.OPENPGP
        assert envelope.signature_bytes == armor + b"\n"
        assert envelope.signed_payload == (
            ("tree %s\nparent %s\nauthor %s\ncommitter %s\n"
             % (tree, parent, ACTOR_LINE, ACTOR_LINE)).encode()
            + b"\nsigned commit\n"
        )
        assert envelope.metadata.issuer_key_id == "42" * 8

        # Caching: re-reading does not change anything.
        assert commit.raw_commit is commit.raw_commit
        assert commit.signatures is commit.signatures
    finally:
        gr.clear()


def test_commit_exposes_signed_mergetag(signed_repo):
    path, tree, parent = signed_repo
    nested_armor = armor_openpgp(openpgp_packet(
        hashed=subpacket(0x10, b"\x77" * 8)))
    tag_body = (
        ("object %s\ntype commit\ntag v9\ntagger %s\n"
         % (parent, ACTOR_LINE)).encode()
        + fold_header(b"gpgsig", nested_armor)
        + b"\nrelease\n"
    )
    body = (
        ("tree %s\nparent %s\nauthor %s\ncommitter %s\n"
         % (tree, parent, ACTOR_LINE, ACTOR_LINE)).encode()
        + fold_header(b"mergetag", tag_body)
        + b"\nmerge\n"
    )
    sha = _write_commit(path, body)

    gr = Git(str(path))
    try:
        commit = gr.get_commit(sha)
        assert commit.signatures == ()
        assert len(commit.merge_tags) == 1

        merge_tag = commit.merge_tags[0]
        assert merge_tag.tag_name == "v9"
        assert merge_tag.tag_bytes == tag_body

        nested = merge_tag.signatures[0]
        assert nested.scope == SignatureScope.MERGE_TAG
        assert nested.metadata.issuer_key_id == "77" * 8
        assert nested.signed_payload == (
            ("object %s\ntype commit\ntag v9\ntagger %s\n"
             % (parent, ACTOR_LINE)).encode()
            + b"\nrelease\n"
        )

        # Both default and custom verification run over nested signatures.
        results = commit.verify_signatures()
        assert len(results) == 1
        assert results[0].status == VerificationStatus.UNVERIFIED

        custom = _RecordingVerifier()
        assert commit.verify_signatures(custom)[0].status == \
            VerificationStatus.VERIFIED
        assert len(custom.seen) == 1
    finally:
        gr.clear()


def test_ordinary_commit_is_unsigned(signed_repo):
    path, _tree, _parent = signed_repo
    head = _git(path, "rev-parse", "HEAD").stdout.decode().strip()

    gr = Git(str(path))
    try:
        commit = gr.get_commit(head)
        assert commit.signatures == ()
        assert commit.merge_tags == ()
        assert commit.verify_signatures() == []
    finally:
        gr.clear()


def test_verification_never_invokes_external_processes(signed_repo, monkeypatch):
    path, tree, parent = signed_repo
    body = (
        ("tree %s\nparent %s\nauthor %s\ncommitter %s\n"
         % (tree, parent, ACTOR_LINE, ACTOR_LINE)).encode()
        + fold_header(b"gpgsig", armor_ssh(ssh_signature_blob()))
        + b"\nssh signed\n"
    )
    sha = _write_commit(path, body)

    def _boom(*args, **kwargs):
        raise AssertionError("verification must not spawn processes")

    monkeypatch.setattr("subprocess.Popen", _boom)
    monkeypatch.setattr("subprocess.run", _boom)

    gr = Git(str(path))
    try:
        commit = gr.get_commit(sha)
        results = commit.verify_signatures()
        assert len(results) == 1
        assert results[0].status == VerificationStatus.UNVERIFIED
        assert results[0].envelope.armor == ArmorKind.SSH
    finally:
        gr.clear()
