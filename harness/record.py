"""Capture format.

A capture is a JSONL file. Record 0 is the header; every subsequent record is one
RTP packet, transmitted or received, in the order the harness observed it.

Design rule: the capture stores raw payloads and raw timestamps and nothing derived.
All metrics are computed offline by analyse.py. This means metric definitions can be
revised, or a reviewer's alternative definition applied, without re-running a test
that costs GPU hours. Never write a derived latency figure into a capture.

Clocks
------
t_mono_ns   CLOCK_MONOTONIC on the measuring host. This is the metric clock: t0 and
            t1 are both taken from it, on the same host, so mouth-to-ear latency
            needs no inter-host clock synchronisation.
t_wall_ns   CLOCK_REALTIME, recorded only so captures can be joined to SUT-side
            OpenTelemetry spans for the component decomposition. Any skew affects
            the decomposition, never the headline metric.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

from . import __version__

SCHEMA_VERSION = "vlh-capture/1"


@dataclass
class CaptureHeader:
    schema: str = SCHEMA_VERSION
    call_id: str = ""
    prompt_id: str = ""
    prompt_sha256: str = ""
    speech_end_sample: int = -1  # in the prompt's own 8 kHz sample domain
    # Where the system's unprompted greeting finished, in the received stream's own sample
    # domain, or 0 if it had none. Response-onset detection must begin after this point:
    # a greeting arrives before t0, is not a response to anything, and would otherwise be
    # detected as one and yield a large negative MRL. Observed, not derived, so it belongs
    # in the capture.
    greeting_end_sample: int = 0
    codec: str = "pcmu"
    ptime_ms: float = 20.0
    sample_rate: int = 8000
    harness_version: str = __version__
    sut_label: str = ""          # free-text SUT identity, e.g. "parakeet-v2+qwen3-8b+chatterbox"
    sut_config_sha256: str = ""  # hash of the SUT config bundle, for reproducibility
    concurrency: int = 1         # number of concurrent calls in the run this call belonged to
    run_id: str = ""
    ground_truth_ms: float | None = None  # populated only by synthetic/reference captures
    notes: dict = field(default_factory=dict)

    @property
    def samples_per_frame(self) -> int:
        return int(round(self.sample_rate * self.ptime_ms / 1000.0))


@dataclass
class Packet:
    dir: str        # "tx" (harness -> SUT) or "rx" (SUT -> harness)
    t_mono_ns: int
    t_wall_ns: int
    seq: int
    rtp_ts: int
    ssrc: int
    pt: int
    marker: bool
    payload: bytes

    def to_json(self) -> dict:
        return {
            "d": self.dir,
            "m": self.t_mono_ns,
            "w": self.t_wall_ns,
            "s": self.seq,
            "r": self.rtp_ts,
            "x": self.ssrc,
            "p": self.pt,
            "k": 1 if self.marker else 0,
            "b": base64.b64encode(self.payload).decode("ascii"),
        }

    @staticmethod
    def from_json(o: dict) -> "Packet":
        return Packet(
            dir=o["d"],
            t_mono_ns=o["m"],
            t_wall_ns=o["w"],
            seq=o["s"],
            rtp_ts=o["r"],
            ssrc=o["x"],
            pt=o["p"],
            marker=bool(o["k"]),
            payload=base64.b64decode(o["b"]),
        )


@dataclass
class Capture:
    header: CaptureHeader
    packets: list[Packet] = field(default_factory=list)

    def tx(self) -> list[Packet]:
        return [p for p in self.packets if p.dir == "tx"]

    def rx(self) -> list[Packet]:
        return [p for p in self.packets if p.dir == "rx"]


def _open_write(path: Path):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "wb"), encoding="utf-8")
    return open(path, "w", encoding="utf-8")


def _open_read(path: Path):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def write_capture(path: str | Path, cap: Capture) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_write(path) as fh:
        fh.write(json.dumps({"header": asdict(cap.header)}) + "\n")
        for p in cap.packets:
            fh.write(json.dumps(p.to_json()) + "\n")
    return path


def read_capture(path: str | Path) -> Capture:
    with _open_read(Path(path)) as fh:
        first = json.loads(fh.readline())
        if "header" not in first:
            raise ValueError("capture is missing its header record")
        hdr = CaptureHeader(**first["header"])
        if hdr.schema != SCHEMA_VERSION:
            raise ValueError(f"unsupported capture schema {hdr.schema!r}")
        pkts = [Packet.from_json(json.loads(line)) for line in fh if line.strip()]
    return Capture(header=hdr, packets=pkts)


def iter_captures(root: str | Path) -> Iterator[Capture]:
    for p in sorted(Path(root).rglob("*.jsonl*")):
        yield read_capture(p)
