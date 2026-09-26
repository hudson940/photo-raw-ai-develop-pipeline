"""Client-side (end-to-end) encryption for object-storage payloads.

When STORAGE_ENCRYPTION_KEY is set, every file is encrypted here before it is
uploaded and decrypted after it is downloaded, so the S3-compatible bucket only
ever holds ciphertext. The local `data/` cache stays plaintext — the pipeline
(rawpy/OpenCV/PIL) needs real files — only the remote object is encrypted.

Format (streaming, authenticated):

    magic "PRENC1" | version 0x01 | 8-byte random nonce-prefix
    then repeated frames:  flag(1) | ct_len(4, big-endian) | ciphertext(ct_len)

Each frame is one AES-256-GCM chunk (<=1 MiB plaintext). The per-chunk nonce is
nonce_prefix || chunk_index(4B), and the frame's flag byte (0x00 normal, 0x01
final) is authenticated as GCM associated data — so truncating, reordering, or
dropping the last chunk is detected on decrypt, not just bit-flips.

Decryption is tolerant of *plaintext* input (no magic header): it copies the
bytes through unchanged. That lets encryption be switched on over a bucket that
already holds unencrypted objects — old files still read, new ones are encrypted.
"""

import hashlib
import logging
import os
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

log = logging.getLogger("crypto")

_MAGIC = b"PRENC1"
_VERSION = b"\x01"
_HEADER = _MAGIC + _VERSION            # 7 bytes
_CHUNK = 1 << 20                       # 1 MiB plaintext per GCM frame
_TAG = 16                             # GCM tag length

# Fixed salt for passphrase stretching. A raw 32-byte key (recommended) skips this
# entirely; the salt only matters when a human passphrase is used, where its job is
# to make the KDF deterministic across restarts (so the same passphrase always
# derives the same key). Override with STORAGE_ENCRYPTION_SALT for a per-deployment salt.
_DEFAULT_SALT = b"photocapibara/storage-enc/v1"


def load_key(secret: str) -> bytes:
    """Turn the configured secret into a 32-byte AES key.

    A 64-char hex string or a base64 value that decodes to exactly 32 bytes is used
    directly; anything else is treated as a passphrase and stretched with scrypt.
    """
    secret = (secret or "").strip()
    if not secret:
        raise ValueError("empty storage encryption key")
    # hex-encoded 32 bytes
    if len(secret) == 64:
        try:
            return bytes.fromhex(secret)
        except ValueError:
            pass
    # base64-encoded 32 bytes
    try:
        import base64
        raw = base64.b64decode(secret, validate=True)
        if len(raw) == 32:
            return raw
    except Exception:
        pass
    # otherwise: a passphrase -> scrypt
    salt = os.environ.get("STORAGE_ENCRYPTION_SALT", "").encode() or _DEFAULT_SALT
    # maxmem must exceed 128*r*n (=32 MiB here); OpenSSL's default cap is 32 MiB.
    return hashlib.scrypt(secret.encode(), salt=salt, n=1 << 15, r=8, p=1,
                          dklen=32, maxmem=64 << 20)


def encrypt_file(src: Path, dst: Path, key: bytes) -> None:
    """Encrypt `src` into `dst` (streamed, chunk by chunk)."""
    aead = AESGCM(key)
    prefix = os.urandom(8)
    src, dst = Path(src), Path(dst)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(_HEADER)
        fout.write(prefix)
        index = 0
        prev = fin.read(_CHUNK)
        while True:
            nxt = fin.read(_CHUNK)
            final = not nxt                       # `prev` is the last chunk
            flag = b"\x01" if final else b"\x00"
            nonce = prefix + index.to_bytes(4, "big")
            ct = aead.encrypt(nonce, prev, flag)
            fout.write(flag)
            fout.write(len(ct).to_bytes(4, "big"))
            fout.write(ct)
            index += 1
            if final:
                break
            prev = nxt


def _read_exactly(f, n: int) -> bytes:
    buf = f.read(n)
    if len(buf) != n:
        raise ValueError("encrypted stream truncated")
    return buf


def decrypt_file(src: Path, dst: Path, key: bytes) -> None:
    """Decrypt `src` into `dst`. If `src` is not our format (no magic header) the
    bytes are copied through unchanged, so plaintext objects still read."""
    src, dst = Path(src), Path(dst)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        head = fin.read(len(_HEADER))
        if head[:len(_MAGIC)] != _MAGIC:          # not encrypted — passthrough
            fout.write(head)
            while chunk := fin.read(_CHUNK):
                fout.write(chunk)
            return
        if head[len(_MAGIC):] != _VERSION:
            raise ValueError("unsupported encryption version")
        aead = AESGCM(key)
        prefix = _read_exactly(fin, 8)
        index = 0
        seen_final = False
        while not seen_final:
            flag = fin.read(1)
            if not flag:
                raise ValueError("encrypted stream truncated (no final chunk)")
            n = int.from_bytes(_read_exactly(fin, 4), "big")
            if n < _TAG or n > _CHUNK + _TAG:
                raise ValueError("corrupt chunk length")
            ct = _read_exactly(fin, n)
            nonce = prefix + index.to_bytes(4, "big")
            try:
                pt = aead.decrypt(nonce, ct, flag)
            except InvalidTag as exc:
                raise ValueError("authentication failed (wrong key or tampered data)") from exc
            fout.write(pt)
            index += 1
            if flag == b"\x01":
                seen_final = True
        if fin.read(1):
            raise ValueError("trailing data after final chunk")
