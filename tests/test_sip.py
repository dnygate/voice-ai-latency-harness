"""SIP signalling correctness: the pure layer.

Separate from test_core.py because these need `sipmessage`, which is an optional extra
required by stage 4 alone. Stages 1 to 3 and everything in test_core.py run on numpy.

Digest responses are checked against hashes computed in the test rather than against
values this code produced, because a test that asserts the implementation agrees with
itself would pass just as happily on a wrong construction.
"""

from __future__ import annotations

import hashlib

import pytest

sipmessage = pytest.importorskip("sipmessage", reason="stage 4 extra: pip install -e '.[sip]'")

from sipmessage import URI, AuthChallenge, Request, Response  # noqa: E402

from harness.sip import (  # noqa: E402
    MediaAnswer, address, build_request, build_sdp_offer, challenge_from,
    digest_credentials, new_branch, parse_sdp_answer,
)


def _md5(s: str) -> str:
    return hashlib.md5(s.encode(), usedforsecurity=False).hexdigest()


# ------------------------------------------------------------------ digest authentication

def test_digest_with_qop_matches_rfc7616_construction():
    ch = AuthChallenge.parse('Digest realm="pbx", nonce="dead", qop="auth", algorithm=MD5')
    cr = digest_credentials(ch, username="1001", password="s3cret", method="REGISTER",
                            uri="sip:pbx", cnonce="0a4f113b", nc=1)
    ha1 = _md5("1001:pbx:s3cret")
    ha2 = _md5("REGISTER:sip:pbx")
    assert cr.parameters.get("response") == _md5(f"{ha1}:dead:00000001:0a4f113b:auth:{ha2}")
    assert cr.parameters.get("nc") == "00000001"
    assert cr.parameters.get("qop") == "auth"
    assert cr.parameters.get("cnonce") == "0a4f113b"


def test_digest_without_qop_takes_the_two_argument_form():
    """RFC 2069. Sending cnonce and nc to a server that offered no qop, or omitting them
    when it did, is answered with a further challenge rather than an error, so both
    branches have to be right and neither is exercised by the other."""
    ch = AuthChallenge.parse('Digest realm="pbx", nonce="dead"')
    cr = digest_credentials(ch, username="u", password="p", method="INVITE", uri="sip:x")
    ha1, ha2 = _md5("u:pbx:p"), _md5("INVITE:sip:x")
    assert cr.parameters.get("response") == _md5(f"{ha1}:dead:{ha2}")
    assert cr.parameters.get("qop") is None
    assert cr.parameters.get("nc") is None
    assert cr.parameters.get("cnonce") is None


def test_digest_sess_variant_rehashes_ha1_with_the_nonces():
    ch = AuthChallenge.parse('Digest realm="pbx", nonce="n1", qop="auth", algorithm=MD5-sess')
    cr = digest_credentials(ch, username="u", password="p", method="INVITE", uri="sip:x",
                            cnonce="cn", nc=1)
    ha1 = _md5(f"{_md5('u:pbx:p')}:n1:cn")
    assert cr.parameters.get("response") == _md5(f"{ha1}:n1:00000001:cn:auth:{_md5('INVITE:sip:x')}")


def test_digest_supports_sha256_because_some_registrars_prefer_it():
    ch = AuthChallenge.parse('Digest realm="pbx", nonce="n", qop="auth", algorithm=SHA-256')
    cr = digest_credentials(ch, username="u", password="p", method="INVITE", uri="sip:x",
                            cnonce="cn", nc=1)
    ha1 = hashlib.sha256(b"u:pbx:p").hexdigest()
    ha2 = hashlib.sha256(b"INVITE:sip:x").hexdigest()
    want = hashlib.sha256(f"{ha1}:n:00000001:cn:auth:{ha2}".encode()).hexdigest()
    assert cr.parameters.get("response") == want


def test_unsupported_digest_algorithm_is_refused_loudly():
    ch = AuthChallenge.parse('Digest realm="r", nonce="n", algorithm=WHIRLPOOL')
    with pytest.raises(ValueError, match="unsupported digest algorithm"):
        digest_credentials(ch, username="u", password="p", method="INVITE", uri="sip:x")


def test_nonce_count_increments_change_the_response():
    """A replayed nc is a replay attack from the registrar's point of view and is
    rejected, so a caller re-authenticating on a new request must advance it."""
    ch = AuthChallenge.parse('Digest realm="r", nonce="n", qop="auth"')
    kw = dict(username="u", password="p", method="INVITE", uri="sip:x", cnonce="cn")
    assert (digest_credentials(ch, nc=1, **kw).parameters.get("response")
            != digest_credentials(ch, nc=2, **kw).parameters.get("response"))


def test_challenge_is_read_from_the_right_header_for_401_and_407():
    """A 401 challenges in WWW-Authenticate and a 407 in Proxy-Authenticate, and the
    answers go back in different headers. Crossing them loops rather than failing."""
    r401 = Response.parse(b'SIP/2.0 401 Unauthorized\r\nWWW-Authenticate: Digest realm="a", nonce="n"\r\n\r\n')
    r407 = Response.parse(b'SIP/2.0 407 Proxy Authentication Required\r\nProxy-Authenticate: Digest realm="b", nonce="n"\r\n\r\n')
    assert challenge_from(r401).parameters.get("realm") == "a"
    assert challenge_from(r401, proxy=True) is None
    assert challenge_from(r407, proxy=True).parameters.get("realm") == "b"
    assert challenge_from(r407) is None


# ------------------------------------------------------------------------------------ SDP

def test_media_level_connection_line_overrides_the_session_level_one():
    """RFC 4566 §5.7. A peer answering from a media gateway gives a different media
    address from its signalling address; taking the session-level line sends audio to the
    far end's SIP interface, where nothing is listening and the call is silent."""
    ans = parse_sdp_answer(
        b"v=0\r\no=- 9 9 IN IP4 203.0.113.1\r\ns=-\r\nc=IN IP4 203.0.113.1\r\nt=0 0\r\n"
        b"m=audio 51000 RTP/AVP 0 101\r\nc=IN IP4 198.51.100.77\r\na=rtpmap:0 PCMU/8000\r\na=ptime:20\r\n")
    assert ans.address == "198.51.100.77"
    assert ans.port == 51000
    assert ans.codec == "pcmu"
    assert ans.ptime_ms == 20.0
    assert not ans.is_held


def test_session_level_connection_is_used_when_media_has_none():
    ans = parse_sdp_answer(b"v=0\r\nc=IN IP4 192.0.2.9\r\nt=0 0\r\nm=audio 6000 RTP/AVP 8\r\n")
    assert ans.address == "192.0.2.9"
    assert ans.codec == "pcma"


def test_zero_port_is_reported_as_held_rather_than_dialled_into():
    """RFC 3264 §8.4. Transmitting into a declined stream produces a capture with no
    reply, which at the analyser is indistinguishable from a system that failed to
    answer. The caller has to be told before it starts sending."""
    assert parse_sdp_answer(b"v=0\r\nc=IN IP4 1.2.3.4\r\nm=audio 0 RTP/AVP 0\r\n").is_held


def test_video_stream_does_not_capture_the_audio_port():
    ans = parse_sdp_answer(
        b"v=0\r\nc=IN IP4 1.2.3.4\r\nm=video 7000 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\n"
        b"m=audio 7002 RTP/AVP 0\r\na=ptime:20\r\n")
    assert ans.port == 7002


def test_answer_with_no_usable_audio_stream_raises():
    with pytest.raises(ValueError, match="no audio stream"):
        parse_sdp_answer(b"v=0\r\no=- 1 1 IN IP4 1.2.3.4\r\ns=-\r\nt=0 0\r\n")


def test_offer_advertises_the_port_it_was_given_and_refuses_unknown_codecs():
    """The media socket is bound before signalling starts, so the port is passed in. A
    port advertised before it is bound can be taken by another process in between."""
    off = build_sdp_offer(address="10.0.0.5", port=40100, codec="pcma", session_id=7).decode()
    assert "m=audio 40100 RTP/AVP 8 101" in off
    assert "a=rtpmap:8 PCMA/8000" in off
    assert "c=IN IP4 10.0.0.5" in off
    assert "a=ptime:20" in off
    with pytest.raises(ValueError, match="not one this harness can encode"):
        build_sdp_offer(address="10.0.0.5", port=1, codec="opus")


def test_offer_and_answer_round_trip_through_our_own_parser():
    off = build_sdp_offer(address="10.0.0.5", port=40100, session_id=1)
    ans = parse_sdp_answer(off)
    assert ans.address == "10.0.0.5" and ans.port == 40100 and ans.codec == "pcmu"


# -------------------------------------------------------------------------- wire format

def _invite(**kw):
    return build_request(
        "INVITE", request_uri=URI(scheme="sip", user="441234", host="trunk.example.net"),
        from_addr=address(URI(scheme="sip", user="1001", host="trunk.example.net"), tag="ft"),
        to_addr=address(URI(scheme="sip", user="441234", host="trunk.example.net")),
        call_id="cid@10.0.0.5", cseq=1, via_host="10.0.0.5", via_port=5060,
        contact=address(URI(scheme="sip", user="1001", host="10.0.0.5", port=5060)),
        body=build_sdp_offer(address="10.0.0.5", port=40100, session_id=1),
        content_type="application/sdp", **kw)


def test_request_serialises_parameters_rather_than_python_reprs():
    """Regression. Handing a plain dict where sipmessage expects Parameters serialises the
    dict's repr, producing `Via: SIP/2.0/UDP host:port{'branch': ...}`, which every SBC
    rejects. It is invisible in the object graph and only shows on the wire."""
    wire = bytes(_invite())
    assert b"{'" not in wire and b'{"' not in wire
    assert b";branch=z9hG4bK" in wire
    assert b";tag=ft" in wire
    assert b";rport" in wire


def test_request_round_trips_and_keeps_every_header_a_uac_needs():
    req = _invite()
    back = Request.parse(bytes(req))
    assert back.method == "INVITE"
    assert back.cseq.sequence == 1 and back.cseq.method == "INVITE"
    assert back.call_id == "cid@10.0.0.5"
    assert back.max_forwards == 70
    assert back.via[0].parameters.get("branch").startswith("z9hG4bK")
    assert back.from_address.parameters.get("tag") == "ft"
    assert back.contact[0].uri.port == 5060
    assert back.body == req.body


def test_branch_carries_the_rfc3261_magic_cookie_and_is_unique_per_transaction():
    """§8.1.1.7. Without the cookie a proxy falls back to RFC 2543 transaction matching,
    where a retransmission can be taken for a new request."""
    branches = {new_branch() for _ in range(64)}
    assert len(branches) == 64
    assert all(b.startswith("z9hG4bK") for b in branches)


def test_credentials_appear_in_authorization_and_survive_reparsing():
    ch = AuthChallenge.parse('Digest realm="pbx", nonce="dead", qop="auth"')
    cr = digest_credentials(ch, username="1001", password="p", method="INVITE",
                            uri="sip:441234@trunk.example.net", cnonce="cn")
    back = Request.parse(bytes(_invite(credentials=cr)))
    assert back.authorization.scheme == "Digest"
    assert back.authorization.parameters.get("realm") == "pbx"
    assert back.authorization.parameters.get("response") == cr.parameters.get("response")


def test_content_length_is_derived_from_the_body_not_asserted():
    """A hand-set Content-Length that disagrees with the body is how a request gets
    silently truncated by a proxy, so the library derives it and we never set it."""
    req = _invite()
    assert Request.parse(bytes(req)).content_length == len(req.body)
