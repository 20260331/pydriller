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
Raw signature envelopes of Git commits.

Supply-chain analyses need the *exact bytes* a cryptographic signature covers.
GitPython parses the ``gpgsig`` header into a re-encoded, newline-stripped
string and skips ``mergetag`` headers altogether, so reconstructing the signed
buffer from the formatted attributes of a Commit is not byte-accurate.

This module reads the raw commit object, parses the folded ``gpgsig`` and
``mergetag`` headers (RFC 4880 folding: continuation lines start with a single
space), and preserves:

* the unfolded signature bytes exactly as stored in the header;
* the *signed payload*, i.e. the raw object body with the signature header
  block removed (the same byte range Git itself strips before verifying);
* offline metadata parsed directly out of OpenPGP (RFC 4880) and SSH
  (PROTOCOL.certkeys) signature packets.

No external ``gpg``/``ssh-keygen`` binary, keyring or network access is
required: the default :class:`OfflineSignatureVerifier` only performs
structural inspection. Cryptographic verification can be plugged in through
the :class:`SignatureVerifier` interface.
"""
import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

logger = logging.getLogger(__name__)

#: Headers that carry a detached signature over the object body. ``gpgsig`` is
#: used by Git for OpenPGP, SSH and X.509 signatures alike; ``gpgsig-sha256``
#: is the SHA-256 object-namespaced variant.
SIGNATURE_HEADERS = frozenset({b"gpgsig", b"gpgsig-sha256"})


class SignatureScope(str, Enum):
    """
    The object a signature covers: the commit itself or an embedded
    (signed) tag reachable through a ``mergetag`` header.
    """

    COMMIT = "commit"
    MERGE_TAG = "mergetag"


class ArmorKind(str, Enum):
    """
    The wire format detected from the ASCII-armor marker of a ``gpgsig``
    header value.
    """

    OPENPGP = "openpgp"
    SSH = "ssh"
    CMS = "cms"
    UNKNOWN = "unknown"


class VerificationStatus(str, Enum):
    """
    Outcome of a signature verification.

    Custom verifiers plugged in through :class:`SignatureVerifier` may return
    any of these states; the offline verifier never returns
    :attr:`VERIFIED` because it holds no keys.
    """

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    MALFORMED = "malformed"


# RFC 4880 section 5.2.1
_PUBLIC_KEY_ALGORITHMS: Dict[int, str] = {
    1: "RSA (Encrypt or Sign)",
    2: "RSA Encrypt-Only",
    3: "RSA Sign-Only",
    16: "ElGamal Encrypt-Only",
    17: "DSA",
    18: "ECDH",
    19: "ECDSA",
    20: "ElGamal Encrypt or Sign",
    21: "Diffie-Hellman",
    22: "EdDSA",
    23: "AEDSA",
}

# RFC 4880 section 9.4
_HASH_ALGORITHMS: Dict[int, str] = {
    1: "MD5",
    2: "SHA-1",
    3: "RIPEMD-160",
    8: "SHA-256",
    9: "SHA-384",
    10: "SHA-512",
    11: "SHA-224",
}

# RFC 4880 section 5.2.1
_SIGNATURE_TYPES: Dict[int, str] = {
    0x00: "Binary document signature",
    0x01: "Canonical text document signature",
    0x02: "Standalone signature",
    0x10: "Generic User ID certification",
    0x11: "Persona User ID certification",
    0x12: "Casual User ID certification",
    0x13: "Positive User ID certification",
    0x18: "Subkey Binding Signature",
    0x19: "Primary Key Binding Signature",
    0x1F: "Signature directly on a Key",
    0x20: "Key revocation signature",
    0x28: "Subkey revocation signature",
    0x30: "Certification revocation signature",
    0x40: "Timestamp signature",
    0x50: "Third-Party Confirmation signature",
}


@dataclass(frozen=True)
class OpenPGPSignatureInfo:
    """
    Metadata extracted offline from an RFC 4880 version 4 signature packet.

    Nothing here proves the signature is valid: the values are parsed from
    unauthenticated packet bytes and only describe how the signature was
    produced.
    """

    version: int
    signature_type: int
    signature_type_name: str
    public_key_algorithm: int
    public_key_algorithm_name: str
    hash_algorithm: int
    hash_algorithm_name: str
    creation_time: Optional[datetime] = None
    issuer_key_id: Optional[str] = None
    issuer_fingerprint: Optional[str] = None


@dataclass(frozen=True)
class SSHSignatureInfo:
    """
    Metadata extracted offline from an ``-----BEGIN SSH SIGNATURE-----``
    blob (see PROTOCOL.certkeys in the OpenSSH sources).
    """

    public_key_type: str
    namespace: str
    hash_algorithm: str
    signature_algorithm: str


@dataclass(frozen=True)
class SignatureEnvelope:
    """
    A single signature header found on a commit or on an embedded tag.

    :ivar scope: object covered by the signature
    :ivar header: header name as stored (``gpgsig`` or ``gpgsig-sha256``)
    :ivar signature_bytes: unfolded header value, byte for byte
    :ivar signed_payload: exact bytes the signature is computed over, i.e.
        the raw object body with this signature header block removed
    :ivar armor: armor format detected from the BEGIN marker
    :ivar metadata: offline packet metadata (OpenPGP/SSH), if parsed
    :ivar parse_error: set when the armor is recognized but its packets
        cannot be parsed
    """

    scope: SignatureScope
    header: str
    signature_bytes: bytes
    signed_payload: bytes
    armor: ArmorKind
    metadata: Optional[Union[OpenPGPSignatureInfo, SSHSignatureInfo]] = None
    parse_error: Optional[str] = None

    @property
    def is_structurally_valid(self) -> bool:
        """:return: True if the signature packets could be parsed offline."""
        return self.parse_error is None


@dataclass(frozen=True)
class MergeTagEnvelope:
    """
    A ``mergetag`` header embedded in a signed merge commit.

    The unfolded value is a complete tag object body; if that tag itself is
    signed (signed annotated tag merged into the commit), its signature
    envelopes are exposed in :attr:`signatures`.

    :ivar target: SHA of the object the tag points at
    :ivar object_type: type of the tagged object (usually ``commit``)
    :ivar tag_name: tag name
    :ivar tagger: raw ``tagger`` header value (actor and date), if present
    :ivar message: tag message bytes
    :ivar tag_bytes: complete unfolded tag object body
    :ivar signatures: signatures covering the embedded tag
    """

    target: str
    object_type: str
    tag_name: str
    tagger: Optional[str]
    message: bytes
    tag_bytes: bytes
    signatures: Tuple[SignatureEnvelope, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VerificationResult:
    """
    Result of verifying one :class:`SignatureEnvelope`.

    This is the pluggable result interface: a custom
    :class:`SignatureVerifier` returns instances of this class (or a
    subclass) with its own trust decisions.
    """

    envelope: SignatureEnvelope
    status: VerificationStatus
    verified: Optional[bool]
    reason: Optional[str] = None
    key_id: Optional[str] = None
    fingerprint: Optional[str] = None
    creation_time: Optional[datetime] = None


class SignatureVerifier(ABC):
    """
    Pluggable interface for signature verification.

    Implement this class to integrate an external verifier (a local GnuPG
    installation, a hardware token, a key-server client, ...). Implementations
    decide how much they trust keys; PyDriller itself ships only the
    keyless :class:`OfflineSignatureVerifier`.
    """

    @abstractmethod
    def verify(self, envelope: SignatureEnvelope) -> VerificationResult:
        """
        Verify a single signature envelope.

        :param envelope: the signature to verify
        :return: the verification decision
        """
        raise NotImplementedError


class OfflineSignatureVerifier(SignatureVerifier):
    """
    Structural, keyless verifier.

    It never shells out to ``gpg``/``ssh-keygen``, never reads a keyring and
    never accesses the network. Signatures whose packets parse correctly are
    reported as :attr:`VerificationStatus.UNVERIFIED` together with the
    metadata extracted from the packets; unparseable signatures are reported
    as :attr:`VerificationStatus.MALFORMED`.
    """

    def verify(self, envelope: SignatureEnvelope) -> VerificationResult:
        if envelope.parse_error is not None:
            return VerificationResult(
                envelope=envelope,
                status=VerificationStatus.MALFORMED,
                verified=False,
                reason=envelope.parse_error,
            )

        key_id = None
        fingerprint = None
        creation_time = None
        if isinstance(envelope.metadata, OpenPGPSignatureInfo):
            key_id = envelope.metadata.issuer_key_id
            fingerprint = envelope.metadata.issuer_fingerprint
            creation_time = envelope.metadata.creation_time

        return VerificationResult(
            envelope=envelope,
            status=VerificationStatus.UNVERIFIED,
            verified=None,
            reason="offline structural inspection only; no cryptographic "
                   "verification performed",
            key_id=key_id,
            fingerprint=fingerprint,
            creation_time=creation_time,
        )


#: Shared default verifier: stateless and safe to reuse.
offline_signature_verifier = OfflineSignatureVerifier()


class _Header(NamedTuple):
    name: bytes
    value: bytes
    start: int
    end: int


def _parse_headers(raw: bytes) -> Tuple[List[_Header], int]:
    """
    Parse the header section of a raw Git object body (commit or tag).

    Folded header values (continuation lines starting with exactly one
    space) are unfolded by removing that single leading space from each
    continuation line, keeping every line terminator. This is the exact
    inverse of Git's strbuf_add_lines() folding (used for gpgsig and
    mergetag), so embedded tag bodies round-trip byte for byte.
    The returned byte ranges cover the whole folded block in ``raw``, so
    deleting ``raw[start:end]`` removes the header exactly as Git's
    remove_signature() does.

    :param raw: raw object body
    :return: headers and the offset at which the message (after the blank
        separator line) begins
    """
    headers: List[_Header] = []
    pos = 0
    size = len(raw)

    while pos < size:
        line_end = raw.find(b"\n", pos)
        if line_end == -1:
            line_end = size
        line = raw[pos:line_end]

        # An empty line separates headers from the message.
        if line == b"":
            return headers, min(pos + 1, size)
        # A continuation without a header means malformed input; stop.
        if line[:1] == b" ":
            break

        space = line.find(b" ")
        if space == -1:
            name, first_value = line, b""
        else:
            name, first_value = line[:space], line[space + 1:]

        block_start = pos
        # Folding prepends a space to complete lines and drops nothing:
        # the first line keeps its terminator, every continuation keeps
        # its own. Restore both so the value round-trips byte for byte.
        value = first_value
        if line_end < size:
            value += b"\n"
        pos = line_end + 1

        while pos < size and raw[pos:pos + 1] == b" ":
            cont_end = raw.find(b"\n", pos)
            if cont_end == -1:
                # Final continuation line without a terminator.
                value += raw[pos + 1:size]
                pos = size
                break
            value += raw[pos + 1:cont_end + 1]
            pos = cont_end + 1

        headers.append(
            _Header(name, value, block_start, pos)
        )

    return headers, pos


def _detect_armor(signature: bytes) -> ArmorKind:
    first_line = signature.split(b"\n", 1)[0].strip()
    if first_line == b"-----BEGIN PGP SIGNATURE-----":
        return ArmorKind.OPENPGP
    if first_line == b"-----BEGIN SSH SIGNATURE-----":
        return ArmorKind.SSH
    if first_line in (
        b"-----BEGIN SIGNED MESSAGE-----",
        b"-----BEGIN CMS-----",
        b"-----BEGIN PKCS7-----",
    ):
        return ArmorKind.CMS
    return ArmorKind.UNKNOWN


def _dearmor(signature: bytes) -> bytes:
    """Decode the base64 body of an ASCII-armored block."""
    lines = signature.splitlines()
    begin = None
    end = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if begin is None and stripped.startswith(b"-----BEGIN "):
            begin = i
        elif begin is not None and stripped.startswith(b"-----END "):
            end = i
            break
    if begin is None or end is None:
        raise ValueError("armor BEGIN/END markers not found")

    body_lines: List[bytes] = []
    in_body = False
    for line in lines[begin + 1:end]:
        stripped = line.strip()
        if not in_body:
            if not stripped:
                in_body = True
                continue
            # Armor headers look like "Key: value"; base64 never contains ":".
            if b":" not in stripped:
                in_body = True
                body_lines.append(stripped)
            continue
        if not stripped or stripped.startswith(b"="):
            # Blank line or Radix-64 checksum line ("=crc24").
            break
        body_lines.append(stripped)

    if not body_lines:
        raise ValueError("empty armor body")
    return base64.b64decode(b"".join(body_lines))


def _u16(data: bytes, offset: int) -> int:
    return (data[offset] << 8) | data[offset + 1]


def _u32(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset:offset + 4], "big")


def _iter_packets(data: bytes) -> List[Tuple[int, bytes]]:
    """Split an OpenPGP binary object into (packet_tag, body) pairs."""
    packets: List[Tuple[int, bytes]] = []
    pos = 0
    size = len(data)

    while pos < size:
        first = data[pos]
        pos += 1
        if not first & 0x80:
            raise ValueError("invalid OpenPGP packet header")

        if first & 0x40:
            # New-format packet length, possibly using partial body lengths.
            tag = first & 0x3F
            body = b""
            while True:
                octet = data[pos]
                if octet < 192:
                    length = octet
                    pos += 1
                elif octet < 224:
                    length = ((octet - 192) << 8) + data[pos + 1] + 192
                    pos += 2
                elif octet == 255:
                    length = _u32(data, pos + 1)
                    pos += 5
                else:
                    # Partial body length: read the chunk and continue.
                    length = 1 << (octet & 0x1F)
                    pos += 1
                    body += data[pos:pos + length]
                    pos += length
                    continue
                body += data[pos:pos + length]
                pos += length
                break
        else:
            tag = (first >> 2) & 0x0F
            length_type = first & 0x03
            if length_type == 0:
                length = data[pos]
                pos += 1
            elif length_type == 1:
                length = _u16(data, pos)
                pos += 2
            elif length_type == 2:
                length = _u32(data, pos)
                pos += 4
            else:
                length = size - pos
            body = data[pos:pos + length]
            pos += length

        packets.append((tag, body))

    return packets


def _parse_subpackets(data: bytes) -> List[Tuple[int, bytes]]:
    subpackets: List[Tuple[int, bytes]] = []
    pos = 0
    while pos < len(data):
        octet = data[pos]
        if octet < 192:
            length = octet
            pos += 1
        elif octet < 255:
            length = ((octet - 192) << 8) + data[pos + 1] + 192
            pos += 2
        else:
            length = _u32(data, pos + 1)
            pos += 5
        body = data[pos:pos + length]
        pos += length
        if len(body) < 1:
            raise ValueError("truncated signature subpacket")
        subpackets.append((body[0] & 0x7F, body[1:]))
    return subpackets


def _parse_openpgp_signature(blob: bytes) -> OpenPGPSignatureInfo:
    """Parse an RFC 4880 version 4 signature packet out of dearmored bytes."""
    packets = _iter_packets(blob)
    try:
        packet_body = next(body for tag, body in packets if tag == 2)
    except StopIteration:
        raise ValueError("no OpenPGP signature packet (tag 2) found")

    if len(packet_body) < 5:
        raise ValueError("truncated signature packet")

    version = packet_body[0]
    if version != 4:
        raise ValueError(
            "unsupported OpenPGP signature version %d (only version 4 is "
            "parsed offline)" % version
        )

    sig_type = packet_body[1]
    pk_algo = packet_body[2]
    hash_algo = packet_body[3]
    pos = 4

    hashed_len = _u16(packet_body, pos)
    pos += 2
    hashed = packet_body[pos:pos + hashed_len]
    if len(hashed) != hashed_len:
        raise ValueError("truncated hashed subpacket area")
    pos += hashed_len

    unhashed_len = _u16(packet_body, pos)
    pos += 2
    unhashed = packet_body[pos:pos + unhashed_len]
    if len(unhashed) != unhashed_len:
        raise ValueError("truncated subpacket area")

    creation_time: Optional[datetime] = None
    issuer_key_id: Optional[str] = None
    issuer_fingerprint: Optional[str] = None

    # Hashed subpackets take precedence over the unhashed area.
    for sub_type, sub_data in _parse_subpackets(hashed) + \
            _parse_subpackets(unhashed):
        if sub_type == 0x02 and len(sub_data) == 4 and creation_time is None:
            creation_time = datetime.fromtimestamp(
                _u32(sub_data, 0), tz=timezone.utc
            )
        elif sub_type == 0x10 and len(sub_data) == 8 and issuer_key_id is None:
            issuer_key_id = sub_data.hex().upper()
        elif sub_type == 0x21 and issuer_fingerprint is None:
            # First octet is the fingerprint version (4, 5 or 6).
            fingerprint = sub_data[1:]
            if fingerprint:
                issuer_fingerprint = fingerprint.hex().upper()

    return OpenPGPSignatureInfo(
        version=version,
        signature_type=sig_type,
        signature_type_name=_SIGNATURE_TYPES.get(sig_type, "Unknown"),
        public_key_algorithm=pk_algo,
        public_key_algorithm_name=_PUBLIC_KEY_ALGORITHMS.get(pk_algo, "Unknown"),
        hash_algorithm=hash_algo,
        hash_algorithm_name=_HASH_ALGORITHMS.get(hash_algo, "Unknown"),
        creation_time=creation_time,
        issuer_key_id=issuer_key_id,
        issuer_fingerprint=issuer_fingerprint,
    )


def _read_ssh_string(data: bytes, offset: int) -> Tuple[bytes, int]:
    length = _u32(data, offset)
    start = offset + 4
    end = start + length
    if end > len(data):
        raise ValueError("truncated SSH signature string")
    return data[start:end], end


def _read_ssh_u32(data: bytes, offset: int) -> Tuple[int, int]:
    return _u32(data, offset), offset + 4


def _parse_ssh_signature(blob: bytes) -> SSHSignatureInfo:
    """Parse an SSH signature wire blob (PROTOCOL.certkeys)."""
    magic, pos = _read_ssh_string(blob, 0)
    if magic != b"SSHSIG":
        raise ValueError("not an SSH signature blob (bad magic)")

    version, pos = _read_ssh_u32(blob, pos)
    if version != 1:
        raise ValueError("unsupported SSH signature version %d" % version)

    public_key_blob, pos = _read_ssh_string(blob, pos)
    public_key_type, _ = _read_ssh_string(public_key_blob, 0)

    namespace, pos = _read_ssh_string(blob, pos)
    _reserved, pos = _read_ssh_string(blob, pos)
    hash_algorithm, pos = _read_ssh_string(blob, pos)
    signature_blob, pos = _read_ssh_string(blob, pos)
    signature_algorithm, _ = _read_ssh_string(signature_blob, 0)

    return SSHSignatureInfo(
        public_key_type=public_key_type.decode("ascii", "replace"),
        namespace=namespace.decode("ascii", "replace"),
        hash_algorithm=hash_algorithm.decode("ascii", "replace"),
        signature_algorithm=signature_algorithm.decode("ascii", "replace"),
    )


def _build_envelope(
    scope: SignatureScope,
    header_name: bytes,
    signature_bytes: bytes,
    signed_payload: bytes,
) -> SignatureEnvelope:
    armor = _detect_armor(signature_bytes)
    metadata: Optional[Union[OpenPGPSignatureInfo, SSHSignatureInfo]] = None
    parse_error: Optional[str] = None

    if armor == ArmorKind.OPENPGP:
        try:
            metadata = _parse_openpgp_signature(_dearmor(signature_bytes))
        except (ValueError, IndexError) as exc:
            parse_error = "malformed OpenPGP signature: %s" % exc
    elif armor == ArmorKind.SSH:
        try:
            metadata = _parse_ssh_signature(_dearmor(signature_bytes))
        except (ValueError, IndexError) as exc:
            parse_error = "malformed SSH signature: %s" % exc
    elif armor == ArmorKind.CMS:
        # X.509/CMS DER is opaque without a crypto library; the raw bytes
        # remain available in signature_bytes for external tooling.
        pass
    else:
        parse_error = "unrecognized signature armor"

    return SignatureEnvelope(
        scope=scope,
        header=header_name.decode("ascii", "replace"),
        signature_bytes=signature_bytes,
        signed_payload=signed_payload,
        armor=armor,
        metadata=metadata,
        parse_error=parse_error,
    )


def parse_tag_envelopes(raw_tag: bytes) -> Tuple[Tuple[SignatureEnvelope, ...], bytes]:
    """
    Parse the signatures of a raw tag object body (standalone or unfolded
    from a ``mergetag`` header).

    :param raw_tag: raw tag object body
    :return: signature envelopes whose signed payload is the tag body with
        the signature header removed, and the tag message bytes
    """
    headers, message_start = _parse_headers(raw_tag)
    signatures: List[SignatureEnvelope] = []
    for header in headers:
        if header.name in SIGNATURE_HEADERS:
            signatures.append(
                _build_envelope(
                    SignatureScope.MERGE_TAG,
                    header.name,
                    header.value,
                    raw_tag[:header.start] + raw_tag[header.end:],
                )
            )
    return tuple(signatures), raw_tag[message_start:]


def _parse_merge_tag(tag_bytes: bytes) -> MergeTagEnvelope:
    headers, message_start = _parse_headers(tag_bytes)
    fields: Dict[bytes, bytes] = {}

    for header in headers:
        fields.setdefault(header.name, header.value)

    signatures, _message = parse_tag_envelopes(tag_bytes)
    tagger = fields.get(b"tagger")

    def _field(name: bytes) -> bytes:
        value = fields.get(name, b"")
        # Scalar tag headers are single lines; the folded value keeps the
        # line terminator, which is not part of the value itself.
        return value[:-1] if value.endswith(b"\n") else value

    return MergeTagEnvelope(
        target=_field(b"object").decode("utf-8", "replace"),
        object_type=_field(b"type").decode("ascii", "replace"),
        tag_name=_field(b"tag").decode("utf-8", "replace"),
        tagger=_field(b"tagger").decode("utf-8", "replace")
        if tagger is not None else None,
        message=tag_bytes[message_start:],
        tag_bytes=tag_bytes,
        signatures=tuple(signatures),
    )


def parse_commit_envelopes(
    raw_commit: bytes,
) -> Tuple[Tuple[SignatureEnvelope, ...], Tuple[MergeTagEnvelope, ...]]:
    """
    Parse the signature-related headers of a raw commit object body.

    :param raw_commit: raw commit object body, as returned by
        ``git cat-file commit <hash>`` or read from the GitPython
        ``data_stream``
    :return: commit-level signature envelopes (usually zero or one) and
        embedded merge-tag envelopes, in header order
    """
    headers, _message_start = _parse_headers(raw_commit)
    signatures: List[SignatureEnvelope] = []
    merge_tags: List[MergeTagEnvelope] = []

    for header in headers:
        if header.name in SIGNATURE_HEADERS:
            signatures.append(
                _build_envelope(
                    SignatureScope.COMMIT,
                    header.name,
                    header.value,
                    raw_commit[:header.start] + raw_commit[header.end:],
                )
            )
        elif header.name == b"mergetag":
            merge_tags.append(_parse_merge_tag(header.value))

    return tuple(signatures), tuple(merge_tags)
