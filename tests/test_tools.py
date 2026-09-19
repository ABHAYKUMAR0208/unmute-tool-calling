"""Tests for the tool registry and HTTP dispatch (fork addition).

These use a stub transport rather than a live backend, so they run in CI
without the voice-ai-agent service. The behaviour they pin down is the one that
keeps the agent honest: every failure path must still yield `ok: false` plus a
`speakable` sentence, and must never look like a success.
"""

import json

import httpx
import pytest

from unmute.llm import tools


def _client_returning(status_code: int, body, request_sink=None):
    """An httpx.AsyncClient factory whose POST returns a canned response."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request_sink is not None:
            request_sink.append(request)
        if isinstance(body, str):
            return httpx.Response(status_code, text=body)
        return httpx.Response(status_code, json=body)

    class _Client(httpx.AsyncClient):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    return _Client


@pytest.fixture
def patch_client(monkeypatch):
    def apply(status_code, body, request_sink=None):
        monkeypatch.setattr(
            httpx, "AsyncClient", _client_returning(status_code, body, request_sink)
        )

    return apply


def test_tool_note_is_one_readable_line():
    note = tools.tool_note(
        "get_service_price",
        {"service": "airport taxi"},
        {"ok": False, "speakable": "We don't quote taxi fares   over the phone."},
    )
    assert note == (
        "get_service_price(service='airport taxi') -> nothing found: "
        "We don't quote taxi fares over the phone."
    )


def test_tool_notes_go_right_after_the_system_prompt():
    messages = [
        {"role": "system", "content": "You are Kelly."},
        {"role": "user", "content": "How much is the biryani?"},
    ]
    out = tools.with_tool_notes(
        messages, ["get_menu(query='biryani') -> found: 320 rupees"]
    )
    assert [m["role"] for m in out] == ["system", "system", "user"]
    assert "get_menu(query='biryani')" in out[1]["content"]
    assert len(messages) == 2  # input list untouched
    assert tools.with_tool_notes(messages, []) is messages


def test_tool_notes_are_capped():
    notes = [f"get_menu(query='{i}') -> found: x" for i in range(40)]
    out = tools.with_tool_notes([{"role": "system", "content": "s"}], notes)
    assert out[1]["content"].count("get_menu(") == tools.MAX_TOOL_NOTES
    assert "query='39'" in out[1]["content"] and "query='0')" not in out[1]["content"]


@pytest.mark.asyncio
async def test_success_envelope_passes_through(patch_client):
    patch_client(
        200,
        {
            "ok": True,
            "speakable": "Your taxi is booked, Raju will collect you at 6 PM.",
            "data": {"booking_id": "A1B2"},
        },
    )
    envelope = await tools.dispatch(
        "book_taxi",
        {"room_number": "305", "destination": "airport", "pickup_time": "6 PM"},
        call_id="call_1",
        api_key="k",
    )
    assert envelope["ok"] is True
    assert "Raju" in envelope["speakable"]
    assert envelope["data"]["booking_id"] == "A1B2"


@pytest.mark.asyncio
async def test_business_failure_is_not_an_exception(patch_client):
    """A failure the backend understands still gives the agent words to say."""
    patch_client(
        200,
        {
            "ok": False,
            "speakable": "I'm sorry, no drivers are free right now.",
            "error_code": "NO_DRIVERS_AVAILABLE",
            "data": {},
        },
    )
    envelope = await tools.dispatch("book_taxi", {"room_number": "305"}, api_key="k")
    assert envelope["ok"] is False
    assert envelope["error_code"] == "NO_DRIVERS_AVAILABLE"
    assert envelope["speakable"]


@pytest.mark.asyncio
async def test_request_shape_matches_contract(patch_client):
    """Headers, URL and idempotency key are what Repo A expects."""
    sink: list[httpx.Request] = []
    patch_client(200, {"ok": True, "speakable": "done", "data": {}}, sink)

    await tools.dispatch(
        "book_taxi",
        {"room_number": "305", "destination": "airport"},
        call_id="call_abc",
        base_url="http://backend:8081",
        api_key="secret",
    )

    request = sink[0]
    assert str(request.url) == "http://backend:8081/internal/tools/book_taxi"
    assert request.headers["X-Tool-Api-Key"] == "secret"
    payload = json.loads(request.content)
    assert payload["room_number"] == "305"
    assert payload["idempotency_key"] == "call_abc-book_taxi-305"


@pytest.mark.asyncio
async def test_auth_rejection_never_reads_as_success(patch_client):
    patch_client(401, {"detail": "bad key"})
    envelope = await tools.dispatch("book_taxi", {"room_number": "305"}, api_key="bad")
    assert envelope["ok"] is False
    assert envelope["error_code"] == "HTTP_401"
    assert envelope["speakable"]


@pytest.mark.asyncio
async def test_non_json_response_is_handled(patch_client):
    patch_client(200, "<html>gateway error</html>")
    envelope = await tools.dispatch("book_taxi", {"room_number": "305"}, api_key="k")
    assert envelope["ok"] is False
    assert envelope["error_code"] == "MALFORMED_RESPONSE"


@pytest.mark.asyncio
async def test_missing_speakable_is_backfilled(patch_client):
    """`speakable` is mandatory; never let the LLM improvise one instead."""
    patch_client(200, {"ok": True, "data": {"booking_id": "X"}})
    envelope = await tools.dispatch("book_taxi", {"room_number": "305"}, api_key="k")
    assert envelope["speakable"]


@pytest.mark.asyncio
async def test_unreachable_backend_yields_speakable(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    class _Client(httpx.AsyncClient):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    envelope = await tools.dispatch("book_taxi", {"room_number": "305"}, api_key="k")
    assert envelope["ok"] is False
    assert envelope["error_code"] == "TOOL_UNREACHABLE"
    assert "sorry" in envelope["speakable"].lower()


@pytest.mark.asyncio
async def test_timeout_yields_speakable(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    class _Client(httpx.AsyncClient):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    envelope = await tools.dispatch("book_taxi", {"room_number": "305"}, api_key="k")
    assert envelope["ok"] is False
    assert envelope["error_code"] == "TOOL_TIMEOUT"
    assert envelope["speakable"]


@pytest.mark.asyncio
async def test_hallucinated_tool_never_leaves_the_fork(patch_client):
    sink: list[httpx.Request] = []
    patch_client(200, {"ok": True, "speakable": "done", "data": {}}, sink)
    envelope = await tools.dispatch("summon_dragon", {}, api_key="k")
    assert envelope["ok"] is False
    assert envelope["error_code"] == "UNKNOWN_TOOL"
    assert sink == []  # no HTTP request was made


def test_idempotency_key_shape():
    key = tools.make_idempotency_key("call_1", "book_taxi", {"room_number": "305"})
    assert key == "call_1-book_taxi-305"


def test_idempotency_key_unique_without_call_id():
    """Two distinct calls must not collide onto one key and get deduped."""
    a = tools.make_idempotency_key("", "book_taxi", {"room_number": "305"})
    b = tools.make_idempotency_key("", "book_taxi", {"room_number": "305"})
    assert a != b


def test_all_six_contract_tools_present():
    contract_tools = {s["function"]["name"] for s in tools.TOOL_SCHEMAS}
    assert contract_tools == {
        "get_guest_by_room",
        "book_taxi",
        "create_food_order",
        "create_laundry_request",
        "create_maintenance_ticket",
        "create_bill_and_payment_link",
    }
    # TOOL_NAMES additionally carries the local smoke-test tool.
    assert contract_tools < tools.TOOL_NAMES


def test_schemas_are_well_formed():
    for schema in tools.TOOL_SCHEMAS:
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["name"] and function["description"]
        params = function["parameters"]
        assert params["type"] == "object"
        for required in params["required"]:
            assert required in params["properties"]


def test_read_only_tool_is_not_marked_side_effecting():
    """get_guest_by_room may be cancelled freely; the rest may not."""
    assert "get_guest_by_room" not in tools.SIDE_EFFECTING_TOOLS
    assert tools.SIDE_EFFECTING_TOOLS < tools.TOOL_NAMES


# --- Local smoke-test tool -------------------------------------------------


@pytest.mark.asyncio
async def test_local_tool_runs_without_any_backend(monkeypatch):
    """The whole point: no HTTP, no key, no Repo A."""

    def explode(*a, **kw):
        raise AssertionError("a local tool must not touch the network")

    monkeypatch.setattr(httpx, "AsyncClient", explode)
    envelope = await tools.dispatch("get_hotel_time", {})
    assert envelope["ok"] is True
    assert envelope["speakable"]
    assert "iso" in envelope["data"]


def test_smoke_tool_is_not_side_effecting():
    assert "get_hotel_time" not in tools.SIDE_EFFECTING_TOOLS


def test_smoke_tool_absent_from_the_real_tool_set():
    """It must never be offered alongside the real hotel tools."""
    assert "get_hotel_time" not in {s["function"]["name"] for s in tools.TOOL_SCHEMAS}


def test_ingredients_only_when_the_guest_asks():
    # Seen live: "I want to eat a burger" got the recipe read out.
    for ordering in ("I want to eat a burger", "just order me a burger", "one brownie please",
                     "Yes, and with no sugar."):
        assert not tools.asked_about_details(ordering), ordering
    for asking in ("what's in the butter chicken?", "does it contain nuts", "I'm allergic to egg",
                   "what extras can I add", "is it spicy", "is the biryani vegetarian"):
        assert tools.asked_about_details(asking), asking
    env = tools.details_not_requested({"item": "burger"})
    assert env["ok"] and env["speakable"] == "Sure, the burger is on our menu."


def test_repeat_lookups_share_one_cache_key():
    k = tools.cache_key
    assert k("get_menu", {"query": "Alcohol"}) == k("get_menu", {"query": "alcoholic beverages"})
    assert k("get_menu", {"query": "  Cold   Coffee "}) == k("get_menu", {"query": "cold coffee"})
    assert k("get_menu", {}) == k("get_menu", {"query": ""}) == k("get_menu", {"query": None})
    assert k("get_menu", {"query": "tea"}) != k("get_menu", {"query": "coffee"})
    assert k("get_menu", {"query": "tea"}) != k("get_service_price", {"service": "tea"})
    assert "submit_guest_requests" not in tools.CACHEABLE_TOOLS  # saving is never cached


def test_cached_result_is_marked_and_the_original_untouched():
    first = {"ok": True, "speakable": "Beer is 350 rupees.", "data": {"price": 350}}
    again = tools.from_cache(first)
    assert again["speakable"] == first["speakable"] and again["data"]["price"] == 350
    assert again["data"]["already_looked_up"] is True
    assert "already_looked_up" not in first["data"]
