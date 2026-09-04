"""SIP signalling for stage 4: the pure layer.

Everything here is a function of its arguments. No sockets, no clocks, no state, so it is
testable without a peer and every branch can be exercised in the suite. Transport,
transactions and dialog state live in `sipcall.py`; this module is what they are built out
of.

Scope, and the boundary that matters
------------------------------------
This module builds and reads SIP messages and SDP. It never constructs an RTP packet,
never opens a media socket, and never sees a timestamp. Media is `loopback.py`'s and it is
the calibrated part of the instrument: pacing measured at 0.002 ms worst deviation across
a qualified host pair, with t0 taken immediately before the packet is handed to the
operating system. Signalling's only media responsibility is to advertise a port it has been
given and to report the address and port the far end answered with.

That boundary is why we write this rather than take a stack. `sipua` would bring aiortc,
whose transceiver owns the media socket; SIPp's own documentation describes it as
"originally a signalling plane traffic generator" with "limited support of media plane
(RTP)" whose pcap replay is "limited to the performances of the system". Both are correct
choices for what they are for. Neither can hold a frame grid to a hundredth of a
millisecond, and surrendering the media path to either would discard stage 3.

Message parsing is `sipmessage`'s, which is a pure-Python package with no runtime
dependencies of its own. It removes the error-prone half of RFC 3261, header folding,
quoted strings, parameter escaping and URI grammar, and leaves us the half we must own.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field

from sipmessage import (URI, Address, AuthChallenge, AuthCredentials, AuthParameters,
                        CSeq, Parameters, Request, Via)

# RTP/AVP static payload types we offer. G.711 only, matching what the harness can encode
# and what METHOD.md validates against frozen golden vectors.
PT_PCMU = 0
PT_PCMA = 8
PT_TELEPHONE_EVENT = 101

_CODEC_PT = {"pcmu": PT_PCMU, "pcma": PT_PCMA}
_PT_NAME = {PT_PCMU: "PCMU", PT_PCMA: "PCMA"}


def new_branch() -> str:
    """A transaction branch. The z9hG4bK prefix is required by RFC 3261 §8.1.1.7 as the
    marker that the sender is RFC 3261 compliant; without it, proxies fall back to the
    RFC 2543 transaction matching rules and a retransmission can be mistaken for a new
    request."""
    return "z9hG4bK" + secrets.token_hex(8)


def new_tag() -> str:
    return secrets.token_hex(6)


def new_call_id(host: str) -> str:
    return f"{secrets.token_hex(10)}@{host}"


def address(uri: URI, *, tag: str | None = None, name: str = "") -> Address:
    """An Address with correctly typed parameters.

    Provided so callers never hand a plain dict to `Address`, which serialises as a Python
    repr rather than as `;tag=...` and yields a request no proxy will accept.
    """
    params = Parameters(tag=tag) if tag is not None else Parameters()
    return Address(uri=uri, name=name, parameters=params)


# --------------------------------------------------------------------------- digest auth

def _H(algorithm: str, data: str) -> str:
    """Hash for the named digest algorithm.

    MD5 is what almost every SBC still offers and RFC 3261 mandates support for. SHA-256
    arrived with RFC 7616 and some modern registrars prefer it, so both are here: being
    offered an algorithm we cannot compute would fail a registration for no good reason.
    MD5 is used here as a challenge-response construction chosen by the peer, not as a
    security primitive we selected.
    """
    algo = algorithm.upper().removesuffix("-SESS")
    if algo in ("", "MD5"):
        h = hashlib.md5(data.encode(), usedforsecurity=False)
    elif algo == "SHA-256":
        h = hashlib.sha256(data.encode())
    elif algo == "SHA-512-256":
        h = hashlib.new("sha512_256", data.encode())
    else:
        raise ValueError(f"unsupported digest algorithm {algorithm!r}")
    return h.hexdigest()


def digest_credentials(challenge: AuthChallenge, *, username: str, password: str,
                       method: str, uri: str, cnonce: str | None = None,
                       nc: int = 1) -> AuthCredentials:
    """Answer a WWW-Authenticate or Proxy-Authenticate challenge.

    RFC 7616 §3.4. Where the challenge offers qop, the client MUST include cnonce and nc
    and hash them into the response; where it does not, the two-argument form of RFC 2069
    applies. Getting that branch wrong produces a second 401 rather than an error, which
    is why both are here and both are tested.
    """
    p = challenge.parameters
    realm = p.get("realm") or ""
    nonce = p.get("nonce") or ""
    algorithm = p.get("algorithm") or "MD5"
    opaque = p.get("opaque")
    qop_offered = [q.strip() for q in (p.get("qop") or "").split(",") if q.strip()]

    ha1 = _H(algorithm, f"{username}:{realm}:{password}")
    if algorithm.upper().endswith("-SESS"):
        # RFC 7616 §3.4.2: the session variant re-hashes with the nonces, so a captured
        # HA1 cannot be replayed against a later nonce.
        ha1 = _H(algorithm, f"{ha1}:{nonce}:{cnonce or ''}")
    ha2 = _H(algorithm, f"{method}:{uri}")

    out = {"username": username, "realm": realm, "nonce": nonce, "uri": uri,
           "algorithm": algorithm}
    if opaque is not None:
        out["opaque"] = opaque

    if qop_offered:
        qop = "auth" if "auth" in qop_offered else qop_offered[0]
        cn = cnonce or secrets.token_hex(8)
        nc_hex = f"{nc:08x}"
        out["response"] = _H(algorithm, f"{ha1}:{nonce}:{nc_hex}:{cn}:{qop}:{ha2}")
        out.update(qop=qop, cnonce=cn, nc=nc_hex)
    else:
        out["response"] = _H(algorithm, f"{ha1}:{nonce}:{ha2}")

    # Keyword form: AuthParameters takes **kwargs, not a mapping.
    return AuthCredentials(scheme="Digest", parameters=AuthParameters(**out))


# --------------------------------------------------------------------------------- SDP

@dataclass
class MediaAnswer:
    """Where the far end wants media sent, and what it agreed to.

    `address` and `port` are the only two values the media path needs from signalling.
    Everything else is recorded so a capture can state what was negotiated rather than
    what was offered.
    """
    address: str
    port: int
    payload_types: list[int] = field(default_factory=list)
    ptime_ms: float | None = None

    @property
    def codec(self) -> str | None:
        for pt in self.payload_types:
            if pt in _PT_NAME:
                return _PT_NAME[pt].lower()
        return None

    @property
    def is_held(self) -> bool:
        """Port zero means the stream is declined or on hold (RFC 3264 §8.4). Sending
        into it would produce a capture with no reply and look like a system that failed
        to answer, so callers must check this rather than discover it at the analyser."""
        return self.port == 0


def build_sdp_offer(*, address: str, port: int, codec: str = "pcmu",
                    ptime_ms: float = 20.0, session_id: int | None = None,
                    with_dtmf: bool = True) -> bytes:
    """An offer for one audio stream on a port the caller already owns.

    The port is passed in rather than chosen here, because the media socket is bound by
    the capture path before signalling starts. That ordering is deliberate: a port
    advertised before it is bound can be taken by another process in between.
    """
    if codec not in _CODEC_PT:
        raise ValueError(f"codec {codec!r} is not one this harness can encode")
    pt = _CODEC_PT[codec]
    sid = session_id if session_id is not None else secrets.randbits(31)
    pts = [pt] + ([PT_TELEPHONE_EVENT] if with_dtmf else [])
    lines = [
        "v=0",
        f"o=- {sid} {sid} IN IP4 {address}",
        "s=-",
        f"c=IN IP4 {address}",
        "t=0 0",
        f"m=audio {port} RTP/AVP " + " ".join(str(p) for p in pts),
        f"a=rtpmap:{pt} {_PT_NAME[pt]}/8000",
    ]
    if with_dtmf:
        # Offered because some platforms require a DTMF path to route a call at all. The
        # harness never sends one: an in-band event would appear in the caller's own audio
        # and shift the annotated speech end.
        lines += [f"a=rtpmap:{PT_TELEPHONE_EVENT} telephone-event/8000",
                  f"a=fmtp:{PT_TELEPHONE_EVENT} 0-15"]
    lines += [f"a=ptime:{ptime_ms:g}", "a=sendrecv"]
    return ("\r\n".join(lines) + "\r\n").encode()


def parse_sdp_answer(body: bytes) -> MediaAnswer:
    """Read the answering SDP for the media address, port and agreed codec.

    A media-level c= line overrides the session-level one (RFC 4566 §5.7), which matters
    on any peer that answers from a media gateway with a different address from its
    signalling. Taking the session-level line unconditionally sends audio to the far end's
    SIP interface, where nothing is listening.
    """
    session_addr: str | None = None
    media_addr: str | None = None
    port: int | None = None
    pts: list[int] = []
    ptime: float | None = None
    in_audio = False

    for raw in body.decode(errors="replace").replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not line or "=" not in line:
            continue
        kind, _, value = line.partition("=")
        if kind == "c":
            parts = value.split()
            if len(parts) >= 3:
                addr_only = parts[2].split("/")[0]
                if in_audio:
                    media_addr = addr_only
                else:
                    session_addr = addr_only
        elif kind == "m":
            parts = value.split()
            in_audio = bool(parts) and parts[0] == "audio"
            if in_audio and len(parts) >= 4:
                port = int(parts[1])
                pts = [int(p) for p in parts[3:] if p.isdigit()]
        elif kind == "a" and in_audio and value.startswith("ptime:"):
            try:
                ptime = float(value.split(":", 1)[1])
            except ValueError:
                pass

    addr = media_addr or session_addr
    if addr is None or port is None:
        raise ValueError("SDP answer has no audio stream with a usable address")
    return MediaAnswer(address=addr, port=port, payload_types=pts, ptime_ms=ptime)


# ---------------------------------------------------------------------- request building

def build_request(method: str, *, request_uri: URI, from_addr: Address, to_addr: Address,
                  call_id: str, cseq: int, via_host: str, via_port: int,
                  contact: Address | None = None, body: bytes = b"",
                  content_type: str | None = None, max_forwards: int = 70,
                  credentials: AuthCredentials | None = None,
                  proxy_credentials: AuthCredentials | None = None,
                  branch: str | None = None, expires: int | None = None,
                  user_agent: str = "voice-ai-latency-harness") -> Request:
    """Assemble a request with the headers RFC 3261 §8.1.1 makes mandatory.

    Content-Length is set explicitly from the body, always, including the zero case.
    `sipmessage` emits no Content-Length of its own, and while §18.3 lets a UDP receiver
    take the body as the remainder of the datagram, §20.14 expects the field present and
    set to zero when there is no body, plenty of SBCs reject requests without it, and it
    becomes mandatory framing the moment the transport is TCP or TLS. Deriving it from
    `len(body)` here means it cannot disagree with what is sent, which is the failure that
    gets a request silently truncated by a proxy.
    """
    req = Request(method=method, uri=request_uri, body=body)
    # Parameters, not a plain dict. A dict serialises as its Python repr, producing a Via
    # of the form `SIP/2.0/UDP host:port{'branch': ...}` that every SBC rejects, and the
    # only way to see it is to look at the bytes on the wire.
    req.via = [Via(transport="UDP", host=via_host, port=via_port,
                   parameters=Parameters(branch=branch or new_branch(), rport=None))]
    req.from_address = from_addr
    req.to_address = to_addr
    req.call_id = call_id
    req.cseq = CSeq(sequence=cseq, method=method)
    req.max_forwards = max_forwards
    req.content_length = len(body)
    req.user_agent = user_agent
    if contact is not None:
        req.contact = [contact]
    if content_type is not None:
        req.content_type = content_type
    if expires is not None:
        req.expires = expires
    if credentials is not None:
        req.authorization = credentials
    if proxy_credentials is not None:
        req.proxy_authorization = proxy_credentials
    return req


def challenge_from(response, *, proxy: bool = False) -> AuthChallenge | None:
    """The challenge in a 401 or 407, or None if there is not one to answer.

    A 401 carries WWW-Authenticate and a 407 carries Proxy-Authenticate, and the answer
    goes back in Authorization or Proxy-Authorization respectively. Crossing them is
    answered with a further challenge rather than an error, which is a tedious thing to
    debug against a live trunk.
    """
    return response.proxy_authenticate if proxy else response.www_authenticate
