"""Preflight: can this machine be trusted to make the measurement?

Run:  python -m harness.preflight

Pacing accuracy is a property of the host, not of the code. A laptop on battery with an
aggressive scheduler will fail where the same code on a tuned server passes easily. This
checks the environment and measures achievable pacing before you invest in a long run,
so a marginal host is identified in ten seconds rather than after an hour of collection.

Stage 1 (`harness.validate`) needs none of this: it derives metrics from synthetically
constructed captures and so is completely insensitive to host timing. Stage 1 is always
runnable. Stage 2 (`harness.loopback`) and anything measuring a real system are not.
"""

from __future__ import annotations

import platform
import sys
import time

import numpy as np

PTIME_GATE_MS = 5.0  # matches the QC blocking threshold in analyse.py


def _pace(n: int, ptime_ms: float, spin: bool) -> np.ndarray:
    """Return absolute deviation from the nominal grid, in ms, for n frame intervals."""
    from .loopback import SPIN_MARGIN_NS, now_ns

    frame_ns = int(ptime_ms * 1e6)
    deadline = now_ns()
    times = []
    for _ in range(n):
        if spin:
            rem = deadline - now_ns() - SPIN_MARGIN_NS
            if rem > 0:
                time.sleep(rem / 1e9)
            while now_ns() < deadline:
                pass
        else:
            rem = deadline - now_ns()
            if rem > 0:
                time.sleep(rem / 1e9)
        times.append(now_ns())
        deadline += frame_ns
    d = np.diff(np.array(times, dtype=np.float64)) / 1e6
    return np.abs(d - ptime_ms)


def _row(label: str, dev: np.ndarray) -> str:
    verdict = "PASS" if dev.max() <= PTIME_GATE_MS else "FAIL"
    return (f"  {label:<28} p50 {np.percentile(dev,50):6.3f}  "
            f"p99 {np.percentile(dev,99):6.3f}  max {dev.max():7.3f} ms   {verdict}")


def main() -> int:
    print("=" * 74)
    print("voice-ai-latency-harness preflight")
    print("=" * 74)
    print(f"  platform     {platform.platform()}")
    print(f"  machine      {platform.machine()}")
    print(f"  python       {platform.python_version()} ({sys.executable})")
    print(f"  numpy        {np.__version__}")
    try:
        import scipy
        print(f"  scipy        {scipy.__version__}  (optional: resampling non-8 kHz prompts)")
    except ImportError:
        print("  scipy        absent  (optional; prompts must then already be 8 kHz mono)")
    try:
        res = time.get_clock_info("monotonic").resolution
        print(f"  monotonic clock resolution  {res * 1e9:.0f} ns")
    except Exception:
        pass

    # audioop is only an oracle for the codec tests and was removed in Python 3.13.
    # The frozen golden vectors replace it, so its absence is not a problem.
    from pathlib import Path
    golden = Path(__file__).resolve().parent.parent / "tests" / "g711_golden.json"
    print(f"  G.711 golden vectors        {'present' if golden.exists() else 'MISSING'}"
          f"  (replaces stdlib audioop, removed in Python 3.13)")

    print("\npacing capability -- can this host hold a frame grid?")
    print("  plain time.sleep() is shown to make the case for the spin; the harness")
    print("  always uses hybrid sleep-then-spin.")
    for ptime in (20.0, 10.0):
        n = int(3000 / ptime)
        print(f"\n  {ptime:g} ms grid, {n} frames:")
        print(_row("plain sleep", _pace(n, ptime, spin=False)))
        dev = _pace(n, ptime, spin=True)
        print(_row("hybrid sleep+spin", dev))

    print("\nverdict")
    dev20 = _pace(150, 20.0, spin=True)
    if dev20.max() <= PTIME_GATE_MS:
        print(f"  This host can pace a 20 ms grid within {dev20.max():.2f} ms. Stage 2 and")
        print("  live measurement are both viable here.")
        ok = True
    else:
        print(f"  This host's worst pacing deviation is {dev20.max():.2f} ms, above the "
              f"{PTIME_GATE_MS:g} ms QC")
        print("  threshold. QC will discard some calls. Stage 1 is unaffected and remains")
        print("  fully usable.")
        ok = False

    if platform.system() == "Darwin":
        print("\nmacOS notes")
        print("  - Battery power and Low Power Mode both degrade pacing substantially.")
        print("    Prefer mains power, and run `caffeinate -di` alongside long collections.")
        print("  - Apple Silicon may schedule the process onto efficiency cores, which")
        print("    hurts pacing. There is no reliable user-space pin, so treat a marginal")
        print("    result as environmental and re-run rather than assuming a code defect.")
        print("  - The spin-wait deliberately burns a core. That is the right trade for")
        print("    accuracy, but it drains battery: avoid long sweeps away from power.")
        print("  - Python 3.13 removed audioop. Nothing here needs it; the golden vectors")
        print("    above are the codec oracle instead.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
