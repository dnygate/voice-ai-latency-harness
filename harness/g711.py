"""G.711 mu-law / A-law codecs, plus L16 passthrough.

Implemented as 65536-entry encode tables and 256-entry decode tables built from a
scalar transcription of the ITU-T G.711 reference algorithm (as in the widely used
g711.c by Sun Microsystems). Table construction is deterministic and happens once
at import; encode/decode are then pure numpy gathers, which keeps the hot path fast
and removes any risk of vectorisation bugs in the bit manipulation.

Do not replace these with audioop: audioop was removed in Python 3.13.
tests/test_g711.py cross-checks against audioop when it is importable.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 8000

_SEG_UEND = (0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF)
_SEG_AEND = (0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF)


def _search(val: int, table: tuple[int, ...]) -> int:
    for i, end in enumerate(table):
        if val <= end:
            return i
    return len(table)


def _lin2ulaw_scalar(pcm: int) -> int:
    pcm >>= 2  # 14-bit
    if pcm < 0:
        pcm = -pcm
        mask = 0x7F
    else:
        mask = 0xFF
    if pcm > 8159:
        pcm = 8159
    pcm += 0x84 >> 2
    seg = _search(pcm, _SEG_UEND)
    if seg >= 8:
        return 0x7F ^ mask
    return ((seg << 4) | ((pcm >> (seg + 1)) & 0x0F)) ^ mask


def _ulaw2lin_scalar(u: int) -> int:
    u = ~u & 0xFF
    t = ((u & 0x0F) << 3) + 0x84
    t <<= (u & 0x70) >> 4
    return (0x84 - t) if (u & 0x80) else (t - 0x84)


def _lin2alaw_scalar(pcm: int) -> int:
    pcm >>= 3  # 13-bit
    if pcm >= 0:
        mask = 0xD5
    else:
        mask = 0x55
        pcm = -pcm - 1
    seg = _search(pcm, _SEG_AEND)
    if seg >= 8:
        return 0x7F ^ mask
    aval = seg << 4
    if seg < 2:
        aval |= (pcm >> 1) & 0x0F
    else:
        aval |= (pcm >> seg) & 0x0F
    return aval ^ mask


def _alaw2lin_scalar(a: int) -> int:
    a ^= 0x55
    t = (a & 0x0F) << 4
    seg = (a & 0x70) >> 4
    if seg == 0:
        t += 8
    elif seg == 1:
        t += 0x108
    else:
        t += 0x108
        t <<= seg - 1
    return t if (a & 0x80) else -t


def _build() -> dict[str, np.ndarray]:
    lin = np.arange(-32768, 32768, dtype=np.int32)
    return {
        "enc_u": np.array([_lin2ulaw_scalar(int(v)) for v in lin], dtype=np.uint8),
        "enc_a": np.array([_lin2alaw_scalar(int(v)) for v in lin], dtype=np.uint8),
        "dec_u": np.array([_ulaw2lin_scalar(i) for i in range(256)], dtype=np.int16),
        "dec_a": np.array([_alaw2lin_scalar(i) for i in range(256)], dtype=np.int16),
    }


_T = _build()

# Codec registry: name -> (bytes per sample, RTP payload type, encode, decode)
_CODECS: dict[str, dict] = {}


def _reg(name: str, pt: int, bps: int, enc, dec) -> None:
    _CODECS[name] = {"pt": pt, "bytes_per_sample": bps, "encode": enc, "decode": dec}


def _enc_u(pcm: np.ndarray) -> bytes:
    return _T["enc_u"][np.asarray(pcm, dtype=np.int16).astype(np.int32) + 32768].tobytes()


def _dec_u(payload: bytes) -> np.ndarray:
    return _T["dec_u"][np.frombuffer(payload, dtype=np.uint8)]


def _enc_a(pcm: np.ndarray) -> bytes:
    return _T["enc_a"][np.asarray(pcm, dtype=np.int16).astype(np.int32) + 32768].tobytes()


def _dec_a(payload: bytes) -> np.ndarray:
    return _T["dec_a"][np.frombuffer(payload, dtype=np.uint8)]


def _enc_l16(pcm: np.ndarray) -> bytes:
    return np.asarray(pcm, dtype=">i2").tobytes()


def _dec_l16(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype=">i2").astype(np.int16)


_reg("pcmu", 0, 1, _enc_u, _dec_u)
_reg("pcma", 8, 1, _enc_a, _dec_a)
_reg("l16", 11, 2, _enc_l16, _dec_l16)  # 8 kHz control codec, no quantisation loss


def codec_names() -> list[str]:
    return sorted(_CODECS)


def payload_type(codec: str) -> int:
    return _CODECS[_norm(codec)]["pt"]


def bytes_per_sample(codec: str) -> int:
    return _CODECS[_norm(codec)]["bytes_per_sample"]


def _norm(codec: str) -> str:
    c = codec.lower()
    if c not in _CODECS:
        raise KeyError(f"unknown codec {codec!r}; known: {codec_names()}")
    return c


def encode(pcm: np.ndarray, codec: str) -> bytes:
    return _CODECS[_norm(codec)]["encode"](pcm)


def decode(payload: bytes, codec: str) -> np.ndarray:
    return _CODECS[_norm(codec)]["decode"](payload)


def roundtrip(pcm: np.ndarray, codec: str) -> np.ndarray:
    """Apply codec quantisation without transport. Used to build degraded eval audio."""
    return decode(encode(pcm, codec), codec)
