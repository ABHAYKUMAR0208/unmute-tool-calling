"""Tests for the local catalogue tools (price, menu, item details)."""

import asyncio

from unmute.llm import catalog, tools


def speak(name, args=None):
    return asyncio.run(tools.dispatch(name, args or {}))


# --- pricing ---------------------------------------------------------------


def test_service_price_exact_hit():
    env = speak("get_service_price", {"service": "swedish massage"})
    assert env["ok"] is True
    assert "2500" in env["speakable"]
    assert env["data"]["price"] == 2500.0


def test_complimentary_service_is_not_quoted_as_zero_rupees():
    env = speak("get_service_price", {"service": "wake up call"})
    assert "complimentary" in env["speakable"]
    assert "0 rupees" not in env["speakable"]


def test_categories_unreachable_in_the_backend_are_reachable_here():
    """gym/pool, business, concierge and kids/pets are not in get_catalog()."""
    for query in ("yoga session", "meeting room", "babysitting", "gift wrapping"):
        assert speak("get_service_price", {"service": query})["ok"] is True


def test_taxi_fares_are_never_quoted():
    from unmute.llm import catalog

    for query in ("airport taxi", "cab to the airport", "airport drop", "taxi", "car hire"):
        env = speak("get_service_price", {"service": query})
        # Answered, not missed: a miss makes the model say "I couldn't find a price".
        assert env["ok"] is True, query
        assert "rupees" not in env["speakable"] and "fare" not in env["speakable"].lower()
        assert "time" in env["speakable"].lower()
        assert env["data"]["fares_quoted"] is False
    assert speak("get_service_price", {"service": "poolside cabana"})["ok"] is True
    assert "cab" not in catalog.SERVICE_CATEGORIES
    assert not hasattr(catalog, "CAB_SERVICES")


def test_unknown_service_is_an_honest_miss():
    env = speak("get_service_price", {"service": "helicopter charter"})
    assert env["ok"] is False
    assert env["error_code"] == "SERVICE_NOT_FOUND"
    assert "rupees" not in env["speakable"]  # never invents a number


# --- menu ------------------------------------------------------------------


def test_menu_with_no_query_lists_sections_not_dishes():
    env = speak("get_menu")
    assert env["ok"] is True
    # 147 dishes must not be read out.
    assert len(env["speakable"]) < 250


def test_menu_results_are_capped():
    env = speak("get_menu", {"query": "biryani"})
    spoken = env["speakable"]
    assert spoken.count(" at ") <= tools.SPOKEN_ITEM_CAP


def test_large_section_reports_a_remainder():
    env = speak("get_menu", {"query": "chinese"})
    assert "more" in env["speakable"]


def test_single_match_gives_a_direct_price():
    env = speak("get_menu", {"query": "cold coffee"})
    assert env["data"]["price"] == 149.0


def test_item_not_on_menu():
    env = speak("get_menu", {"query": "sushi"})
    assert env["ok"] is False
    assert env["error_code"] == "ITEM_NOT_ON_MENU"


# --- item details ----------------------------------------------------------


def test_known_ingredients_are_reported_with_an_allergy_caveat():
    env = speak("get_menu_item_details", {"item": "butter chicken"})
    assert env["ok"] is True
    assert "cashew" in env["speakable"]
    assert "allergy" in env["speakable"].lower()
    assert env["data"]["ingredients_known"] is True


def test_unknown_ingredients_are_never_invented():
    """The drafted table is partial; a gap must be admitted, not filled in."""
    missing = next(n for n in catalog.FOOD_MENU if n not in catalog.INGREDIENTS)
    env = speak("get_menu_item_details", {"item": missing})
    assert env["data"]["ingredients_known"] is False
    assert env["data"]["ingredients"] is None
    assert "do not have the ingredients" in env["speakable"]


def test_no_upsell_when_ingredients_are_unknown():
    missing = next(n for n in catalog.FOOD_MENU if n not in catalog.INGREDIENTS)
    assert (
        "extras" not in speak("get_menu_item_details", {"item": missing})["speakable"]
    )


def test_addons_suit_the_dish():
    coffee = speak("get_menu_item_details", {"item": "cold coffee"})["speakable"]
    assert "extra shot" in coffee
    assert "extra gravy" not in coffee  # nonsense on a drink

    dessert = speak("get_menu_item_details", {"item": "gulab jamun"})["speakable"]
    assert "extra gravy" not in dessert


def test_ambiguous_item_asks_rather_than_picking():
    env = speak("get_menu_item_details", {"item": "tikka"})
    assert env["ok"] is False
    assert env["error_code"] == "AMBIGUOUS_ITEM"
    assert "Did you mean" in env["speakable"]


# --- catalogue integrity ---------------------------------------------------


def test_every_suggested_addon_actually_exists():
    """A rule must never offer an extra that is not in the catalogue."""
    for dish in catalog.FOOD_MENU:
        for addon in catalog.addons_for(dish):
            assert addon in catalog.ADDONS


def test_catalogue_tools_are_read_only():
    for name in ("get_service_price", "get_menu", "get_menu_item_details"):
        assert name not in tools.SIDE_EFFECTING_TOOLS


def test_no_network_for_local_tools(monkeypatch):
    import httpx

    def explode(*a, **kw):
        raise AssertionError("catalogue lookups must not touch the network")

    monkeypatch.setattr(httpx, "AsyncClient", explode)
    assert speak("get_menu", {"query": "burger"})["ok"] is True


def test_drinks_are_found_the_way_guests_ask():
    # Seen live: "alcoholic beverages" found nothing, and "drinks" offered only soft drinks.
    for query in ("alcoholic beverages", "alcohol", "liquor", "hard drinks", "spirits"):
        env = speak("get_menu", {"query": query})
        assert env["ok"] and "beer" in env["speakable"] and "whiskey" in env["speakable"], query
    env = speak("get_menu", {"query": "drinks"})
    assert env["ok"] and "alcoholic drinks" in env["speakable"] and "soft drinks" in env["speakable"]
    assert "Whiskey peg is" in speak("get_menu", {"query": "whisky"})["speakable"]
    assert speak("get_menu", {"query": "soda"})["speakable"] == "Soda is 60 rupees."
