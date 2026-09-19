"""Tool schemas and HTTP dispatch for live tool calling (fork addition).

This module is the fork's ENTIRE knowledge of tools. It deliberately contains:
  - the OpenAI-format schemas the LLM reads (contract section 3), and
  - `dispatch()`, which HTTP-POSTs to the voice-ai-agent backend.

It contains NO business logic, NO pricing, and NO third-party secrets. Booking,
catalog lookups, payment links and the PayU/MSG91/SendGrid keys all live in the
backend (contract section 4.4). If you find yourself adding a price or a
confirmation sentence here, it belongs in the backend instead.

The backend owns truth: every response carries a `speakable` sentence, and the
agent says that rather than inventing a confirmation of its own (section 2).
"""

import asyncio
import re
import json
import os
import uuid
from datetime import datetime
from logging import getLogger
from typing import Any

import httpx

from unmute.kyutai_constants import (
    TOOL_API_BASE_URL,
    TOOL_API_KEY,
    TOOL_SMOKE_TEST,
    TOOL_TIMEOUT_S,
)
from unmute.llm import catalog

logger = getLogger(__name__)

# Tools that change the world. A barge-in must not abort these mid-flight, or a
# guest could be charged for an order the agent never confirmed (see the
# asyncio.shield in the handler, Step 5).
SIDE_EFFECTING_TOOLS = frozenset(
    {
        "book_taxi",
        "create_food_order",
        "create_laundry_request",
        "create_maintenance_ticket",
        "create_bill_and_payment_link",
    }
)

# Spoken when the backend cannot be reached at all. The agent must never claim
# success for a request that never landed, so these are honest non-confirmations.
_TRANSPORT_FAILURE_SPEAKABLE = (
    "I'm sorry, I couldn't complete that just now. "
    "Shall I have the front desk follow up with you?"
)
_TIMEOUT_SPEAKABLE = (
    "I'm sorry, that's taking longer than expected. "
    "Let me have the front desk confirm this for you."
)


def _room_number_property() -> dict[str, Any]:
    return {
        "type": "string",
        "description": "The guest's room number, e.g. '305'.",
    }


def _items_property(example: str) -> dict[str, Any]:
    return {
        "type": "array",
        "description": f"The items requested, e.g. {example}.",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Item name."},
                "quantity": {
                    "type": "integer",
                    "description": "How many, at least 1.",
                    "minimum": 1,
                },
            },
            "required": ["name", "quantity"],
        },
    }


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# The schemas handed to the LLM. Descriptions are prompt surface: they are what
# makes the model pick the right tool and ask for a missing room number rather
# than guessing one.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    _fn(
        "get_guest_by_room",
        "Look up the guest staying in a room, to confirm who you are speaking to. "
        "Read-only: this changes nothing.",
        {"room_number": _room_number_property()},
        ["room_number"],
    ),
    _fn(
        "book_taxi",
        "Book a taxi for a guest. Only call this once you know the room number, "
        "the destination and the pickup time; ask the guest for anything missing.",
        {
            "room_number": _room_number_property(),
            "destination": {
                "type": "string",
                "description": "Where the taxi should go, e.g. 'airport'.",
            },
            "pickup_time": {
                "type": "string",
                "description": (
                    "When to be picked up, as the guest said it, e.g. '6 PM'."
                ),
            },
        },
        ["room_number", "destination", "pickup_time"],
    ),
    _fn(
        "create_food_order",
        "Place a room-service food order. Do not state or estimate a price: the "
        "total is calculated by the hotel system and returned to you.",
        {
            "room_number": _room_number_property(),
            "items": _items_property("two burgers and a pizza"),
            "special_notes": {
                "type": "string",
                "description": "Any special requests, e.g. 'no onions'.",
            },
        },
        ["room_number", "items"],
    ),
    _fn(
        "create_laundry_request",
        "Request a laundry pickup for a guest.",
        {
            "room_number": _room_number_property(),
            "items": _items_property("three shirts"),
            "pickup_time": {
                "type": "string",
                "description": "When to collect the laundry, e.g. '5 PM'.",
            },
        },
        ["room_number", "items"],
    ),
    _fn(
        "create_maintenance_ticket",
        "Report a maintenance problem in a guest's room.",
        {
            "room_number": _room_number_property(),
            "issue_description": {
                "type": "string",
                "description": "The problem, e.g. 'AC not cooling'.",
            },
            "urgency": {
                "type": "string",
                "description": "How urgent the issue is.",
                "enum": ["low", "normal", "high"],
            },
        },
        ["room_number", "issue_description"],
    ),
    _fn(
        "create_bill_and_payment_link",
        "Create a bill for a guest and get a payment link. Do not state or "
        "estimate a total: the hotel system calculates it.",
        {
            "room_number": _room_number_property(),
            "items": _items_property("two burgers"),
        },
        ["room_number", "items"],
    ),
]

# --- Local smoke-test tool -------------------------------------------------
# Executes IN-PROCESS: no backend, no HTTP, no API key. It exists so the whole
# pipeline (STT -> LLM decides -> pause -> execute -> resume -> TTS speaks the
# result) can be demonstrated before Repo A's routes exist.
#
# The time is a good probe precisely because the model cannot know it: if the
# agent says the right time, the round-trip provably happened. Contrast with
# "add two numbers", which a model can answer without calling anything.
LOCAL_TOOLS: dict[str, Any] = {}

SMOKE_TEST_SCHEMAS: list[dict[str, Any]] = [
    _fn(
        "get_hotel_time",
        "Get the current local time at the hotel. Call this whenever the guest "
        "asks what time it is; you have no other way to know it.",
        {},
        [],
    ),
]


def _get_hotel_time(args: dict[str, Any]) -> dict[str, Any]:
    """Local implementation of the smoke-test tool."""
    now = datetime.now()
    return _envelope(
        True,
        f"It's {now.strftime('%-I:%M %p').lower()} here at the hotel."
        if os.name != "nt"
        else f"It's {now.strftime('%I:%M %p').lstrip('0').lower()} here at the hotel.",
        data={"iso": now.isoformat(timespec="seconds")},
    )


LOCAL_TOOLS["get_hotel_time"] = _get_hotel_time


# --- Catalogue tools (local, read-only) ------------------------------------
# Price and menu lookups that run IN-PROCESS against unmute/llm/catalog.py.
# Read-only, so they are safe to cancel on a barge-in and are deliberately
# absent from SIDE_EFFECTING_TOOLS. Backend HTTP routes replace them later.
#
# Answers are SPOKEN, so a 147-item menu cannot be read out. Results are capped
# at SPOKEN_ITEM_CAP and the remainder reported as a count, so the agent lists a
# handful and invites the guest to narrow down.
SPOKEN_ITEM_CAP = 6


def _and_join(names: list[str]) -> str:
    """Join for speech: 'a, b and c'."""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _capped(names: list[str], noun: str) -> str:
    """Speak up to the cap, then say how many more there are."""
    shown = _and_join(names[:SPOKEN_ITEM_CAP])
    extra = len(names) - SPOKEN_ITEM_CAP
    if extra > 0:
        return f"{shown}, and {extra} more {noun}"
    return shown


CATALOG_SCHEMAS: list[dict[str, Any]] = [
    _fn(
        "get_service_price",
        "Look up what a hotel service costs: spa treatments, laundry, room "
        "cleaning, meeting rooms, babysitting and so on. Use this for anything "
        "that is NOT food or drink; for those use get_menu. Taxi and cab fares "
        "are not quoted, so don't look them up. Never quote a price you have not "
        "looked up.",
        {
            "service": {
                "type": "string",
                "description": (
                    "What the guest asked about, in their own words, e.g. "
                    "'swedish massage', 'deep cleaning', 'shirt wash'."
                ),
            }
        },
        ["service"],
    ),
    _fn(
        "get_menu",
        "Look up food and drink: what is available and what it costs. Pass the "
        "guest's words as `query` to search for a dish, or leave it out to hear "
        "the sections. Never quote a price you have not looked up.",
        {
            "query": {
                "type": "string",
                "description": (
                    "A dish, drink or cuisine the guest mentioned, e.g. "
                    "'biryani', 'chinese', 'coffee'. Omit to browse sections."
                ),
            }
        },
        [],
    ),
    _fn(
        "get_menu_item_details",
        "Get the ingredients and available add-ons for one specific dish. ONLY "
        "when the guest asks what is in it, asks about an allergy, or asks which "
        "extras they can have. Never call it just because they're ordering the "
        "dish, and not to check a request like 'no sugar' (take that as a note). "
        "Ingredients are not recorded for every dish and the tool will say so; "
        "never guess what a dish contains.",
        {
            "item": {
                "type": "string",
                "description": "The dish name, e.g. 'butter chicken'.",
            }
        },
        ["item"],
    ),
]


# Whole words only: "cab" must not match "poolside cabana".
_TAXI_PATTERN = re.compile(
    r"\b(taxis?|cabs?|car hire|drivers?|airport (drop|pickup|transfer)|uber)\b"
)


def _is_taxi_query(query: str) -> bool:
    return bool(_TAXI_PATTERN.search(query.lower()))


def _get_service_price(args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("service", "")).strip()
    if not query:
        return _envelope(
            False,
            "Which service did you want the price for?",
            error_code="MISSING_FIELD",
        )

    if _is_taxi_query(query):
        # Taxis are arranged by assigning a driver for the guest's time; fares
        # are not quoted on the phone.
        # ok=True on purpose: a miss makes the model announce "I couldn't find a
        # price for a taxi", which is not what the guest asked for. This is a
        # real answer -- arrange a driver -- with the fare line kept in data for
        # the case where the guest actually asked what it costs.
        return _envelope(
            True,
            "I can arrange a driver for whatever time suits you. When would you "
            "like the taxi, and where are you headed?",
            data={
                "fares_quoted": False,
                "if_the_guest_asked_the_price": (
                    "Taxi fares aren't quoted over the phone."
                ),
                "next_step": "Ask when they want the taxi and where they're going.",
            },
        )

    hits = catalog.find_service(query)
    if not hits:
        # Do not guess: an invented price is worse than an honest miss.
        return _envelope(
            False,
            f"I do not have a price listed for {query}. "
            "Shall I check with the front desk for you?",
            error_code="SERVICE_NOT_FOUND",
        )

    if len(hits) == 1:
        name, price, group = hits[0]
        if price == 0:
            speakable = f"{name.capitalize()} is complimentary."
        else:
            speakable = f"{name.capitalize()} is {catalog.rupees(price)}."
        return _envelope(
            True, speakable, data={"name": name, "price": price, "group": group}
        )

    parts = [f"{n} at {catalog.rupees(p)}" for n, p, _ in hits]
    return _envelope(
        True,
        f"We have {_capped(parts, 'options')}.",
        data={"matches": [{"name": n, "price": p, "group": g} for n, p, g in hits]},
    )


# How guests ask for drinks vs how the menu labels them. Seen live: "alcoholic
# beverages" found nothing and "drinks" matched only the soft-drinks section,
# so a guest asking about alcohol was offered coke and soda.
_ALCOHOL_WORDS = re.compile(
    r"\b(alcohol\w*|liquors?|booze|spirits?|hard drinks?|bar menu|pegs?|drinks? with alcohol)\b"
)
_ALL_DRINKS_WORDS = re.compile(r"^(drinks?|beverages?|something to drink|the drinks menu|drinks menu)$")
_SPELLINGS = {"whisky": "whiskey", "whiskies": "whiskey", "beers": "beer", "wines": "wine"}


def _normalise_menu_query(query: str) -> str:
    q = query.lower().strip()
    for said, menu in _SPELLINGS.items():
        q = re.sub(rf"\b{said}\b", menu, q)
    if _ALCOHOL_WORDS.search(q):
        return "alcoholic"
    return q


def _get_menu(args: dict[str, Any]) -> dict[str, Any]:
    query = _normalise_menu_query(str(args.get("query") or ""))

    if _ALL_DRINKS_WORDS.match(query):
        sections = [
            label.split(":", 1)[1].strip()
            for label in catalog.food_categories()
            if label.lower().startswith("beverages:")
        ]
        # "hot" / "alcoholic" read oddly on their own: "hot drinks".
        sections = [f"{s} drinks" if " " not in s else s for s in sections]
        return _envelope(
            True,
            f"For drinks we have {_capped(sections, 'kinds')}. Which would you like to hear about?",
            data={"categories": sections},
        )

    if not query:
        # No query: describe the sections rather than 147 dishes.
        sections = list(catalog.food_categories())
        return _envelope(
            True,
            f"We have {_capped(sections, 'sections')}. What sounds good?",
            data={"categories": sections},
        )

    hits = catalog.find_food(query)

    if not hits:
        # They may have named a cuisine or section rather than a dish.
        matching = {
            label: names
            for label, names in catalog.food_categories().items()
            if query.lower() in label.lower()
        }
        if matching:
            names = [n for group in matching.values() for n in group]
            return _envelope(
                True,
                f"In that section we have {_capped(names, 'dishes')}.",
                data={"items": names},
            )
        return _envelope(
            False,
            f"I am not finding {query} on the menu. "
            "Would you like me to tell you what we do have?",
            error_code="ITEM_NOT_ON_MENU",
        )

    if len(hits) == 1:
        name, price, category = hits[0]
        return _envelope(
            True,
            f"{name.capitalize()} is {catalog.rupees(price)}.",
            data={"name": name, "price": price, "category": category},
        )

    parts = [f"{n} at {catalog.rupees(p)}" for n, p, _ in hits]
    return _envelope(
        True,
        f"We have {_capped(parts, 'options')}.",
        data={"matches": [{"name": n, "price": p} for n, p, _ in hits]},
    )


def _get_menu_item_details(args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("item", "")).strip()
    if not query:
        return _envelope(False, "Which dish did you mean?", error_code="MISSING_FIELD")

    hits = catalog.find_food(query)
    if not hits:
        return _envelope(
            False,
            f"I am not finding {query} on the menu.",
            error_code="ITEM_NOT_ON_MENU",
        )
    if len(hits) > 1 and hits[0][0].lower() != query.lower():
        names = [n for n, _, _ in hits[:SPOKEN_ITEM_CAP]]
        return _envelope(
            False,
            f"Did you mean {_and_join(names)}?",
            error_code="AMBIGUOUS_ITEM",
            data={"matches": names},
        )

    name, price, _category = hits[0]
    ingredients = catalog.ingredients_for(name)
    addons = catalog.find_addons()

    if ingredients:
        speakable = f"The {name} has {ingredients}."
        if catalog.has_allergen_hint(ingredients):
            # The list is drafted, not the kitchen's own, so it must never
            # stand as an allergy clearance.
            speakable += (
                " If you have an allergy, let me check with the kitchen to be certain."
            )
    else:
        # Absent from the drafted table: say so rather than inventing a list.
        speakable = (
            f"I do not have the ingredients listed for {name}. "
            "I can check with the kitchen if you like."
        )

    # Only pitch extras once we have actually answered the question. Following
    # "I do not know what is in it" with an upsell reads badly.
    suggested = catalog.addons_for(name, limit=SPOKEN_ITEM_CAP)
    if suggested and ingredients:
        speakable += f" For extras, you can add {_and_join(suggested)}."

    return _envelope(
        True,
        speakable,
        data={
            "name": name,
            "price": price,
            "ingredients": ingredients,
            "ingredients_known": ingredients is not None,
            "addons": [{"name": n, "price": p} for n, p in addons],
        },
    )


LOCAL_TOOLS["get_service_price"] = _get_service_price
LOCAL_TOOLS["get_menu"] = _get_menu
LOCAL_TOOLS["get_menu_item_details"] = _get_menu_item_details

TOOL_NAMES = frozenset(
    schema["function"]["name"]  # type: ignore[index]
    for schema in TOOL_SCHEMAS
) | frozenset(LOCAL_TOOLS)


def default_tool_schemas() -> list[dict[str, Any]]:
    """The tool set used when a session doesn't specify its own (Step 6).

    With TOOL_SMOKE_TEST=1 the agent is offered ONLY the local smoke-test tool,
    so the pipeline can be demonstrated end-to-end with no backend running.
    Unset it to go back to the six real hotel tools.
    """
    if TOOL_SMOKE_TEST:
        return [dict(s) for s in SMOKE_TEST_SCHEMAS + CATALOG_SCHEMAS]
    # The catalogue tools are local and read-only, so they are offered
    # alongside the backend tools rather than instead of them.
    return [dict(s) for s in TOOL_SCHEMAS + CATALOG_SCHEMAS]


# --- Tool memory across turns -------------------------------------------
# Tool messages live only in each turn's local message list (Strategy 1), so on
# the next turn the model has no record of what it already looked up and would
# repeat the same lookup. The handler keeps one line per call and passes them
# back in as a system note -- never into chatbot.chat_history.
MAX_TOOL_NOTES = 15


def tool_note(name: str, args: dict[str, Any], envelope: dict[str, Any]) -> str:
    """One line describing a finished tool call, e.g.
    get_menu(query='biryani') -> found: We have veg biryani at 320 rupees."""
    shown_args = ", ".join(
        f"{k}={v!r}" for k, v in args.items() if v not in (None, "", [], {})
    )
    outcome = "found" if envelope.get("ok") else "nothing found"
    speakable = " ".join(str(envelope.get("speakable") or "").split())
    if len(speakable) > 160:
        speakable = speakable[:157] + "..."
    return f"{name}({shown_args}) -> {outcome}: {speakable}"


def with_tool_notes(
    messages: list[dict[str, Any]], notes: list[str]
) -> list[dict[str, Any]]:
    """Insert the earlier lookups as a system note right after the system prompt."""
    if not notes:
        return messages
    note = {
        "role": "system",
        "content": (
            "Lookups you already did earlier in this call. Use these results "
            "instead of calling the same tool again for the same thing, and "
            "don't repeat what you already told the guest:\n- "
            + "\n- ".join(notes[-MAX_TOOL_NOTES:])
        ),
    }
    if messages and messages[0].get("role") == "system":
        return [messages[0], note, *messages[1:]]
    return [note, *messages]


# --- Same lookup twice in one call ------------------------------------------
# A read-only lookup gives the same answer all call long, so a repeat is
# answered from the first result: the guest hears the same thing, and a
# repeated room check doesn't go back to the CRM. Saving requests is never
# cached. The key is normalised so "Alcohol" and "alcoholic beverages" count
# as one lookup.
CACHEABLE_TOOLS = frozenset(
    {"get_menu", "get_service_price", "get_menu_item_details", "verify_room_guest"}
)


def cache_key(name: str, args: dict[str, Any]) -> str:
    normalised: dict[str, Any] = {}
    for key, value in sorted(args.items()):
        if isinstance(value, str):
            value = " ".join(value.lower().split())
            if name == "get_menu" and key == "query":
                value = _normalise_menu_query(value)
        if value not in (None, "", [], {}):
            normalised[key] = value
    return f"{name}:{json.dumps(normalised, sort_keys=True)}"


def from_cache(envelope: dict[str, Any]) -> dict[str, Any]:
    """The earlier result, marked so the model answers from it and moves on."""
    cached = dict(envelope)
    cached["data"] = {
        **(envelope.get("data") or {}),
        "already_looked_up": True,
        "note": (
            "You already looked this up earlier in this call. Answer from this "
            "result, and don't look it up again."
        ),
    }
    return cached


# --- Ingredients only when asked --------------------------------------------
# Llama calls get_menu_item_details on "I want a burger" despite the prompt and
# the tool description, and the guest then hears the recipe. The handler checks
# the guest's own words before running it (see UnmuteHandler._run_tool_calls).
DETAILS_TOOL = "get_menu_item_details"
_DETAIL_QUESTION = re.compile(
    r"\b(what'?s in|what is in|what goes in|in it|inside|ingredients?|made (of|with|from)|"
    r"contains?|allerg\w*|extras?|add[- ]?ons?|toppings?|add |with it|comes? with|spicy|"
    r"veg(an|etarian)?|non[- ]veg|egg|nuts?|peanut|gluten|dairy|lactose|jain|sugar[- ]free)\b"
)


def asked_about_details(user_text: str) -> bool:
    """Did the guest ask what's in a dish, about an allergy, or about extras?"""
    return bool(_DETAIL_QUESTION.search(user_text.lower()))


def details_not_requested(args: dict[str, Any]) -> dict[str, Any]:
    """Stands in for get_menu_item_details when the guest is just ordering."""
    item = str(args.get("item") or "that").strip()
    return _envelope(
        True,
        f"Sure, the {item} is on our menu.",
        data={
            "note": (
                "The guest didn't ask what's in it, so don't list ingredients "
                "or extras. Carry on taking the order."
            )
        },
    )


def make_idempotency_key(call_id: str, name: str, args: dict[str, Any]) -> str:
    """Build the section 4.1 key so a retry cannot double-book or double-charge.

    Contract shape is `<call_id>-<tool>-<room>`. `call_id` comes from the model
    and is not guaranteed unique or even present, so fall back to a random one
    rather than risk two different bookings colliding on the same key.
    """
    room = str(args.get("room_number", "unknown"))
    return f"{call_id or uuid.uuid4().hex}-{name}-{room}"


def _envelope(
    ok: bool,
    speakable: str,
    error_code: str | None = None,
    data: dict | None = None,
) -> dict[str, Any]:
    envelope: dict[str, Any] = {"ok": ok, "speakable": speakable, "data": data or {}}
    if error_code is not None:
        envelope["error_code"] = error_code
    return envelope


async def dispatch(
    name: str,
    args: dict[str, Any],
    call_id: str = "",
    timeout_s: float = TOOL_TIMEOUT_S,
    base_url: str = TOOL_API_BASE_URL,
    api_key: str | None = TOOL_API_KEY,
) -> dict[str, Any]:
    """POST a tool call to the backend and return the contract envelope.

    Always returns an envelope with a `speakable` sentence, even when the
    backend is unreachable, so the agent always has honest words to say. Never
    raises for a business failure; the backend signals those with `ok: false`.
    """
    if name not in TOOL_NAMES:
        # The model hallucinated a tool. Don't say it "failed" -- from the
        # guest's point of view nothing was ever attempted.
        logger.warning("LLM called unknown tool %r", name)
        return _envelope(
            False,
            "I'm sorry, I can't help with that one. Is there something else I can do?",
            error_code="UNKNOWN_TOOL",
        )

    if name in LOCAL_TOOLS:
        # Runs in-process: no backend, no HTTP, no key. Used for the pipeline
        # smoke test before Repo A exists.
        logger.info("Local tool %r (no HTTP)", name)
        try:
            return LOCAL_TOOLS[name](args)
        except Exception:
            logger.exception("Local tool %r raised", name)
            return _envelope(
                False, _TRANSPORT_FAILURE_SPEAKABLE, error_code="LOCAL_TOOL_ERROR"
            )

    payload = dict(args)
    payload["idempotency_key"] = make_idempotency_key(call_id, name, args)
    url = f"{base_url}/internal/tools/{name}"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Tool-Api-Key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(url, json=payload, headers=headers)
    except asyncio.CancelledError:
        raise
    except httpx.TimeoutException:
        logger.warning("Tool %r timed out after %.1fs", name, timeout_s)
        return _envelope(False, _TIMEOUT_SPEAKABLE, error_code="TOOL_TIMEOUT")
    except httpx.HTTPError as exc:
        logger.warning("Tool %r transport error: %s", name, exc)
        return _envelope(
            False, _TRANSPORT_FAILURE_SPEAKABLE, error_code="TOOL_UNREACHABLE"
        )

    if response.status_code != 200:
        # Section 2: non-200 is a transport/auth/validation fault, not a
        # business outcome, so the backend gives us no sentence to speak.
        logger.error(
            "Tool %r returned HTTP %d: %s",
            name,
            response.status_code,
            response.text[:200],
        )
        return _envelope(
            False,
            _TRANSPORT_FAILURE_SPEAKABLE,
            error_code=f"HTTP_{response.status_code}",
        )

    try:
        envelope = response.json()
    except (json.JSONDecodeError, ValueError):
        logger.error("Tool %r returned non-JSON: %s", name, response.text[:200])
        return _envelope(
            False, _TRANSPORT_FAILURE_SPEAKABLE, error_code="MALFORMED_RESPONSE"
        )

    if not isinstance(envelope, dict) or "ok" not in envelope:
        logger.error("Tool %r returned a non-contract envelope: %r", name, envelope)
        return _envelope(
            False, _TRANSPORT_FAILURE_SPEAKABLE, error_code="MALFORMED_RESPONSE"
        )

    if not envelope.get("speakable"):
        # `speakable` is mandatory (sections 2 and 4.5). Rather than let the LLM
        # improvise a confirmation, supply a neutral honest sentence.
        logger.error("Tool %r response has no 'speakable' field", name)
        envelope = dict(envelope)
        envelope["speakable"] = (
            "That's been noted." if envelope.get("ok") else _TRANSPORT_FAILURE_SPEAKABLE
        )

    return envelope
