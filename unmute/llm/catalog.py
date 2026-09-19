"""Hotel service and menu catalogue (fork addition).

Hardcoded on purpose for now: these tools run in-process so the agent can quote
prices and describe the menu before the voice-ai-agent backend exists. When that
backend lands, this file is replaced by HTTP lookups and can be deleted.

Derived from the backend's `service_catalog.py`. Two deliberate differences:

  - That file has two overlapping add-on dicts (ADD_ON_SERVICES and ADDONS)
    which disagree on two prices. ADDONS is the one used here.
  - Five of its categories (gym/pool, business, concierge, kids/pets, add-ons)
    are unreachable through its own get_catalog(). They are included here.

INGREDIENTS ARE UNVERIFIED. They were drafted, not supplied by the kitchen, and
only cover common dishes. Anything absent is reported as unknown rather than
guessed: a guest with an allergy must never be told a dish is safe on the
strength of a drafted list. See `INGREDIENTS` below.
"""

# name -> (price, category). Category drives the browse-by-section answers.
FOOD_MENU: dict[str, tuple[float, str]] = {
    # House specials
    "cheese pizza": (349.0, "House specials"),
    "margherita pizza": (329.0, "House specials"),
    "pepperoni pizza": (399.0, "House specials"),
    "burger": (199.0, "House specials"),
    "chicken burger": (249.0, "House specials"),
    "veg burger": (189.0, "House specials"),
    "sandwich": (149.0, "House specials"),
    "club sandwich": (199.0, "House specials"),
    "grilled sandwich": (169.0, "House specials"),
    "cold coffee": (149.0, "House specials"),
    # Indian: starters
    "paneer tikka": (350.0, "Indian: starters"),
    "chicken tikka": (380.0, "Indian: starters"),
    "malai tikka": (400.0, "Indian: starters"),
    "seekh kebab": (380.0, "Indian: starters"),
    "hara bhara kebab": (280.0, "Indian: starters"),
    "chicken 65": (320.0, "Indian: starters"),
    "amritsari fish": (400.0, "Indian: starters"),
    "papdi chaat": (150.0, "Indian: starters"),
    "samosa": (80.0, "Indian: starters"),
    "pakora": (120.0, "Indian: starters"),
    # Indian: mains
    "butter chicken": (450.0, "Indian: mains"),
    "chicken tikka masala": (460.0, "Indian: mains"),
    "dal makhani": (280.0, "Indian: mains"),
    "dal tadka": (220.0, "Indian: mains"),
    "paneer butter masala": (340.0, "Indian: mains"),
    "palak paneer": (320.0, "Indian: mains"),
    "kadai paneer": (340.0, "Indian: mains"),
    "shahi paneer": (350.0, "Indian: mains"),
    "malai kofta": (330.0, "Indian: mains"),
    "chana masala": (240.0, "Indian: mains"),
    "rajma": (230.0, "Indian: mains"),
    "mixed vegetable curry": (260.0, "Indian: mains"),
    "egg curry": (260.0, "Indian: mains"),
    "mutton curry": (480.0, "Indian: mains"),
    "fish curry": (460.0, "Indian: mains"),
    "prawn curry": (500.0, "Indian: mains"),
    # Indian: rice & biryani
    "chicken biryani": (380.0, "Indian: rice & biryani"),
    "mutton biryani": (450.0, "Indian: rice & biryani"),
    "veg biryani": (320.0, "Indian: rice & biryani"),
    "egg biryani": (300.0, "Indian: rice & biryani"),
    "hyderabadi biryani": (420.0, "Indian: rice & biryani"),
    "jeera rice": (180.0, "Indian: rice & biryani"),
    "lemon rice": (160.0, "Indian: rice & biryani"),
    "curd rice": (150.0, "Indian: rice & biryani"),
    "rice": (120.0, "Indian: rice & biryani"),
    "pulao": (220.0, "Indian: rice & biryani"),
    # Indian: chinese-indian
    "manchurian": (220.0, "Indian: chinese-indian"),
    "veg manchurian": (220.0, "Indian: chinese-indian"),
    "gobi manchurian": (220.0, "Indian: chinese-indian"),
    "chicken manchurian": (260.0, "Indian: chinese-indian"),
    "chilli paneer": (260.0, "Indian: chinese-indian"),
    "chilli chicken": (280.0, "Indian: chinese-indian"),
    # Indian: breads
    "naan": (60.0, "Indian: breads"),
    "butter naan": (70.0, "Indian: breads"),
    "garlic naan": (80.0, "Indian: breads"),
    "roti": (40.0, "Indian: breads"),
    "butter roti": (50.0, "Indian: breads"),
    "tandoori roti": (45.0, "Indian: breads"),
    "paratha": (60.0, "Indian: breads"),
    "aloo paratha": (90.0, "Indian: breads"),
    "kulcha": (60.0, "Indian: breads"),
    "bhatura": (60.0, "Indian: breads"),
    # Indian: accompaniments
    "raita": (80.0, "Indian: accompaniments"),
    "papad": (40.0, "Indian: accompaniments"),
    "pickle": (30.0, "Indian: accompaniments"),
    "salad": (220.0, "Indian: accompaniments"),
    # Chinese
    "veg fried rice": (220.0, "Chinese"),
    "chicken fried rice": (260.0, "Chinese"),
    "egg fried rice": (240.0, "Chinese"),
    "schezwan fried rice": (260.0, "Chinese"),
    "hakka noodles": (220.0, "Chinese"),
    "chicken noodles": (260.0, "Chinese"),
    "schezwan noodles": (260.0, "Chinese"),
    "spring rolls": (200.0, "Chinese"),
    "chicken lollipop": (320.0, "Chinese"),
    "sweet corn soup": (160.0, "Chinese"),
    "hot and sour soup": (170.0, "Chinese"),
    "manchow soup": (170.0, "Chinese"),
    "wonton soup": (190.0, "Chinese"),
    "dim sum": (280.0, "Chinese"),
    "kung pao chicken": (340.0, "Chinese"),
    # Mexican
    "chicken tacos": (320.0, "Mexican"),
    "veg tacos": (280.0, "Mexican"),
    "chicken burrito": (340.0, "Mexican"),
    "veg burrito": (300.0, "Mexican"),
    "nachos": (280.0, "Mexican"),
    "chicken quesadilla": (320.0, "Mexican"),
    "veg quesadilla": (280.0, "Mexican"),
    "guacamole": (200.0, "Mexican"),
    "fajita": (350.0, "Mexican"),
    "enchiladas": (320.0, "Mexican"),
    # Continental / Italian
    "pasta alfredo": (320.0, "Continental / Italian"),
    "pasta arrabbiata": (300.0, "Continental / Italian"),
    "spaghetti bolognese": (340.0, "Continental / Italian"),
    "penne pasta": (300.0, "Continental / Italian"),
    "lasagna": (380.0, "Continental / Italian"),
    "risotto": (360.0, "Continental / Italian"),
    "grilled chicken": (380.0, "Continental / Italian"),
    "grilled fish": (420.0, "Continental / Italian"),
    "caesar salad": (260.0, "Continental / Italian"),
    "greek salad": (240.0, "Continental / Italian"),
    "bruschetta": (220.0, "Continental / Italian"),
    "garlic bread": (150.0, "Continental / Italian"),
    "mushroom soup": (180.0, "Continental / Italian"),
    "tomato soup": (160.0, "Continental / Italian"),
    # Starters (general)
    "soup": (180.0, "Starters (general)"),
    "french fries": (200.0, "Starters (general)"),
    "onion rings": (180.0, "Starters (general)"),
    # Beverages - soft drinks & water
    "coke": (80.0, "Beverages: soft drinks & water"),
    "pepsi": (80.0, "Beverages: soft drinks & water"),
    "sprite": (80.0, "Beverages: soft drinks & water"),
    "soda": (60.0, "Beverages: soft drinks & water"),
    "water bottle": (40.0, "Beverages: soft drinks & water"),
    # Beverages - juices & shakes
    "fresh juice": (150.0, "Beverages: juices & shakes"),
    "orange juice": (150.0, "Beverages: juices & shakes"),
    "watermelon juice": (130.0, "Beverages: juices & shakes"),
    "mango shake": (180.0, "Beverages: juices & shakes"),
    "chocolate shake": (190.0, "Beverages: juices & shakes"),
    "vanilla shake": (180.0, "Beverages: juices & shakes"),
    "lassi": (120.0, "Beverages: juices & shakes"),
    "buttermilk": (60.0, "Beverages: juices & shakes"),
    "mocktail": (220.0, "Beverages: juices & shakes"),
    # Beverages - hot
    "tea": (60.0, "Beverages: hot"),
    "masala chai": (70.0, "Beverages: hot"),
    "green tea": (80.0, "Beverages: hot"),
    "coffee": (100.0, "Beverages: hot"),
    "cappuccino": (150.0, "Beverages: hot"),
    "latte": (160.0, "Beverages: hot"),
    "espresso": (140.0, "Beverages: hot"),
    "hot chocolate": (170.0, "Beverages: hot"),
    # Beverages - alcoholic
    "beer": (350.0, "Beverages: alcoholic"),
    "wine glass": (500.0, "Beverages: alcoholic"),
    "whiskey peg": (450.0, "Beverages: alcoholic"),
    "vodka peg": (400.0, "Beverages: alcoholic"),
    "rum peg": (380.0, "Beverages: alcoholic"),
    "cocktail": (450.0, "Beverages: alcoholic"),
    # Desserts
    "gulab jamun": (120.0, "Desserts"),
    "rasmalai": (150.0, "Desserts"),
    "jalebi": (100.0, "Desserts"),
    "kheer": (130.0, "Desserts"),
    "ice cream": (180.0, "Desserts"),
    "brownie": (200.0, "Desserts"),
    "chocolate cake": (250.0, "Desserts"),
    "cheesecake": (280.0, "Desserts"),
    "tiramisu": (300.0, "Desserts"),
    "fruit salad": (150.0, "Desserts"),
    "pastry": (180.0, "Desserts"),
}

# Add-on name -> price. Sourced from the backend's ADDONS dict.
ADDONS: dict[str, float] = {
    # Cheese / dairy
    "extra cheese": 40.0,
    "extra butter": 20.0,
    "extra paneer": 60.0,
    "extra cream": 30.0,
    # Protein
    "extra chicken": 80.0,
    "extra egg": 20.0,
    "extra prawn": 100.0,
    # Sauce / gravy / spice level
    "extra sauce": 20.0,
    "extra gravy": 30.0,
    "extra masala": 30.0,
    "extra mayo": 20.0,
    "extra ketchup": 0.0,
    "extra spicy": 0.0,
    "extra mild": 0.0,
    "less spicy": 0.0,
    "no spice": 0.0,
    "no onion": 0.0,
    "no garlic": 0.0,
    # Sweetness
    "extra sweet": 0.0,
    "extra sweetness": 0.0,
    "extra sugar": 0.0,
    "no sugar": 0.0,
    "less sweet": 0.0,
    # Veggies / misc
    "extra veggies": 30.0,
    "extra vegetables": 30.0,
    "extra ice": 0.0,
    "extra shot": 40.0,
}

ROOM_SERVICES: dict[str, float] = {
    "basic cleaning": 200.0,
    "deep cleaning": 500.0,
    "turndown service": 0.0,
    "towel replacement": 0.0,
    "bedsheet change": 0.0,
    "minibar refill": 300.0,
    "extra pillow": 0.0,
    "extra blanket": 0.0,
    "extra towel": 0.0,
    "bed making": 100.0,
    "room fragrance service": 150.0,
    "iron and ironing board": 0.0,
}

# Taxi / cab fares are deliberately NOT in the catalogue: taxis are arranged
# by assigning a driver for the guest's chosen time, and fares are never quoted
# on the phone. get_service_price answers taxi questions with that, not a price.

RESTAURANT_SERVICES: dict[str, float] = {
    "table reservation": 0.0,
    "private dining": 2000.0,
    "rooftop dining": 2500.0,
    "candlelight dinner setup": 3000.0,
    "birthday decoration": 1500.0,
    "anniversary decoration": 2000.0,
    "cake 1kg": 800.0,
    "cake half kg": 450.0,
    "live bbq setup": 3500.0,
    "buffet per person": 900.0,
}

LAUNDRY_SERVICES: dict[str, float] = {
    "shirt wash": 80.0,
    "trouser wash": 100.0,
    "suit dry clean": 400.0,
    "saree dry clean": 350.0,
    "kurta dry clean": 250.0,
    "jacket dry clean": 350.0,
    "blanket dry clean": 500.0,
    "express service": 200.0,
    "same day laundry": 300.0,
    "ironing per item": 40.0,
}

SPA_SERVICES: dict[str, float] = {
    "swedish massage 60min": 2500.0,
    "thai massage 60min": 2800.0,
    "deep tissue massage 60min": 3000.0,
    "aromatherapy massage 60min": 2700.0,
    "hot stone massage 60min": 3200.0,
    "couples massage 60min": 5000.0,
    "facial": 1500.0,
    "gold facial": 2200.0,
    "manicure": 800.0,
    "pedicure": 800.0,
    "mani pedi combo": 1400.0,
    "steam bath": 500.0,
    "sauna session": 500.0,
    "hair spa": 1800.0,
}

GYM_POOL_SERVICES: dict[str, float] = {
    "personal training session": 1000.0,
    "yoga session": 700.0,
    "pool towel rental": 100.0,
    "swimming lesson": 1200.0,
    "poolside cabana": 1500.0,
}

BUSINESS_SERVICES: dict[str, float] = {
    "meeting room per hour": 2000.0,
    "conference hall half day": 15000.0,
    "conference hall full day": 25000.0,
    "printing per page": 10.0,
    "scanning per page": 10.0,
    "projector rental": 1500.0,
    "video conferencing setup": 2500.0,
}

CONCIERGE_SERVICES: dict[str, float] = {
    "luggage storage": 0.0,
    "wake up call": 0.0,
    "newspaper delivery": 0.0,
    "sightseeing package half day": 2000.0,
    "sightseeing package full day": 3500.0,
    "event ticket booking assistance": 0.0,
    "courier service": 250.0,
    "flower bouquet delivery": 800.0,
    "gift wrapping": 100.0,
}

KIDS_PET_SERVICES: dict[str, float] = {
    "babysitting per hour": 400.0,
    "kids play area access": 300.0,
    "pet sitting per hour": 350.0,
    "pet food service": 250.0,
}

# Service group -> (human label, catalogue). Order matters only for listing.
SERVICE_CATEGORIES: dict[str, tuple[str, dict[str, float]]] = {
    "room": ("Room", ROOM_SERVICES),
    "restaurant": ("Restaurant", RESTAURANT_SERVICES),
    "laundry": ("Laundry", LAUNDRY_SERVICES),
    "spa": ("Spa", SPA_SERVICES),
    "gym_pool": ("Gym & pool", GYM_POOL_SERVICES),
    "business": ("Business", BUSINESS_SERVICES),
    "concierge": ("Concierge", CONCIERGE_SERVICES),
    "kids_pet": ("Kids & pets", KIDS_PET_SERVICES),
}


# --- DRAFTED INGREDIENTS - NEEDS KITCHEN REVIEW ----------------------------
# These were written from common recipes, NOT supplied by the hotel. They cover
# frequently-asked dishes only; everything else is deliberately absent.
#
# A missing entry is reported as unknown. The tools never infer ingredients from
# a dish name, because a guest with an allergy could act on the answer. Treat
# this whole table as provisional until the kitchen signs it off.
INGREDIENTS: dict[str, str] = {
    # House specials
    "cheese pizza": "wheat base, tomato sauce, mozzarella, herbs",
    "margherita pizza": "wheat base, tomato sauce, mozzarella, fresh basil",
    "pepperoni pizza": "wheat base, tomato sauce, mozzarella, pork pepperoni",
    "burger": "wheat bun, potato patty, lettuce, tomato, onion, mayonnaise",
    "chicken burger": "wheat bun, chicken patty, lettuce, tomato, mayonnaise",
    "veg burger": "wheat bun, mixed vegetable patty, lettuce, tomato, mayonnaise",
    "club sandwich": "white bread, chicken, egg, lettuce, tomato, mayonnaise",
    "grilled sandwich": "white bread, cheese, potato, onion, capsicum, butter",
    "cold coffee": "milk, coffee, sugar, ice cream",
    # Indian starters
    "paneer tikka": "paneer, yoghurt, cream, capsicum, onion, tandoori spices",
    "chicken tikka": "chicken, yoghurt, ginger-garlic, tandoori spices",
    "hara bhara kebab": "spinach, green peas, potato, cashew, gram flour",
    "samosa": "wheat pastry, potato, green peas, cumin, coriander",
    "pakora": "gram flour, onion, potato, green chilli, spices",
    # Indian mains
    "butter chicken": "chicken, tomato, butter, cream, cashew, garam masala",
    "dal makhani": "black lentils, kidney beans, butter, cream, tomato",
    "dal tadka": "yellow lentils, tomato, onion, cumin, ghee",
    "paneer butter masala": "paneer, tomato, butter, cream, cashew, spices",
    "palak paneer": "paneer, spinach, onion, tomato, cream, spices",
    "kadai paneer": "paneer, capsicum, onion, tomato, kadai masala",
    "malai kofta": "potato-paneer dumplings, cashew, cream, tomato gravy",
    "chana masala": "chickpeas, onion, tomato, ginger, garam masala",
    "rajma": "kidney beans, onion, tomato, ginger-garlic, spices",
    # Rice & biryani
    "chicken biryani": "basmati rice, chicken, yoghurt, fried onion, saffron, whole spices",
    "mutton biryani": "basmati rice, mutton, yoghurt, fried onion, saffron, whole spices",
    "veg biryani": "basmati rice, mixed vegetables, yoghurt, fried onion, whole spices",
    "jeera rice": "basmati rice, cumin, ghee",
    # Breads
    "naan": "refined wheat flour, yoghurt, milk, yeast",
    "butter naan": "refined wheat flour, yoghurt, milk, yeast, butter",
    "garlic naan": "refined wheat flour, yoghurt, milk, yeast, garlic, coriander",
    "roti": "whole wheat flour, water",
    "aloo paratha": "whole wheat flour, potato, coriander, spices, ghee",
    # Chinese
    "veg fried rice": "rice, mixed vegetables, soy sauce, spring onion",
    "chicken fried rice": "rice, chicken, egg, soy sauce, spring onion",
    "hakka noodles": "wheat noodles, mixed vegetables, soy sauce, garlic",
    "spring rolls": "wheat wrapper, cabbage, carrot, beans, soy sauce",
    "sweet corn soup": "sweet corn, vegetable stock, cornflour",
    "hot and sour soup": "vegetable stock, cabbage, carrot, soy sauce, vinegar, white pepper",
    # Continental / Italian
    "pasta alfredo": "wheat pasta, cream, butter, parmesan, garlic",
    "pasta arrabbiata": "wheat pasta, tomato, garlic, red chilli, olive oil",
    "spaghetti bolognese": "wheat spaghetti, minced meat, tomato, onion, herbs",
    "lasagna": "wheat pasta sheets, minced meat, tomato, bechamel, cheese",
    "caesar salad": "romaine lettuce, parmesan, croutons, egg-based dressing, anchovy",
    "greek salad": "cucumber, tomato, olives, red onion, feta, olive oil",
    "garlic bread": "wheat bread, garlic, butter, herbs",
    "mushroom soup": "mushroom, cream, butter, vegetable stock",
    "tomato soup": "tomato, cream, butter, herbs",
    # Sides
    "french fries": "potato, salt, vegetable oil",
    "onion rings": "onion, wheat batter, breadcrumbs",
    # Desserts
    "gulab jamun": "milk solids, refined flour, sugar syrup, cardamom",
    "rasmalai": "milk solids, milk, sugar, cardamom, pistachio",
    "kheer": "milk, rice, sugar, cardamom, nuts",
    "ice cream": "milk, cream, sugar, flavouring",
    "brownie": "wheat flour, chocolate, butter, egg, sugar, walnut",
    "chocolate cake": "wheat flour, chocolate, butter, egg, sugar",
    "cheesecake": "cream cheese, biscuit base, butter, sugar, egg",
    "tiramisu": "mascarpone, coffee, egg, savoiardi biscuits, cocoa",
}

# Allergen-bearing words we know appear in the drafted list above. Used only to
# decide whether to add a caveat, never to claim a dish is allergen-free.
_ALLERGEN_HINTS = (
    "wheat",
    "milk",
    "cream",
    "butter",
    "cheese",
    "paneer",
    "egg",
    "cashew",
    "nut",
    "walnut",
    "pistachio",
    "soy",
    "anchovy",
    "yoghurt",
    "mascarpone",
)


def find_food(query: str) -> list[tuple[str, float, str]]:
    """Return (name, price, category) for food matching `query`.

    Longer names are checked first so "chicken biryani" beats a bare "rice",
    matching the convention documented in the backend's own catalogue.
    """
    q = query.lower().strip()
    if not q:
        return []
    ordered = sorted(FOOD_MENU.items(), key=lambda kv: len(kv[0]), reverse=True)
    exact = [(n, p, c) for n, (p, c) in ordered if n == q]
    if exact:
        return exact
    return [(n, p, c) for n, (p, c) in ordered if q in n or n in q]


def find_service(query: str) -> list[tuple[str, float, str]]:
    """Return (name, price, group label) for non-food services matching `query`."""
    q = query.lower().strip()
    if not q:
        return []
    hits: list[tuple[str, float, str]] = []
    for label, catalog in SERVICE_CATEGORIES.values():
        for name, price in catalog.items():
            if name == q:
                return [(name, price, label)]
            if q in name or name in q:
                hits.append((name, price, label))
    return sorted(hits, key=lambda h: len(h[0]))


def find_addons(query: str = "") -> list[tuple[str, float]]:
    """Return (name, price) add-ons, optionally filtered by `query`."""
    q = query.lower().strip()
    items = sorted(ADDONS.items())
    if not q:
        return items
    return [(n, p) for n, p in items if q in n or n in q]


# Which add-ons make sense for which kind of dish. Offering "extra gravy" on a
# dessert reads as a bug to a guest, so add-ons are filtered by the dish rather
# than read off the top of an alphabetical list.
_ADDON_RULES: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    # (words in the dish name, add-ons that suit it)
    (
        ("tea", "coffee", "latte", "cappuccino", "espresso", "chai"),
        ("extra shot", "extra sugar", "no sugar", "less sweet", "extra cream"),
    ),
    (
        ("shake", "juice", "lassi", "mocktail", "soda", "buttermilk"),
        ("extra ice", "extra sugar", "no sugar", "less sweet"),
    ),
    (
        (
            "gulab",
            "rasmalai",
            "jalebi",
            "kheer",
            "ice cream",
            "brownie",
            "cake",
            "cheesecake",
            "tiramisu",
            "pastry",
            "fruit salad",
        ),
        ("extra sweet", "less sweet", "extra cream", "no sugar"),
    ),
    (
        ("pizza", "pasta", "lasagna", "risotto", "sandwich", "burger", "garlic bread"),
        ("extra cheese", "extra veggies", "extra chicken", "extra sauce", "extra mayo"),
    ),
    (
        ("naan", "roti", "paratha", "kulcha", "bhatura"),
        ("extra butter", "extra cheese"),
    ),
]

# Anything savoury that matched no rule above.
_DEFAULT_ADDONS: tuple[str, ...] = (
    "extra cheese",
    "extra chicken",
    "extra paneer",
    "extra gravy",
    "extra spicy",
    "less spicy",
    "no onion",
)


def addons_for(dish: str, limit: int = 6) -> list[str]:
    """Add-ons that suit this dish, most relevant first.

    Only names present in ADDONS are returned, so a rule can never invent an
    extra the hotel does not actually offer.
    """
    name = dish.lower()
    chosen: tuple[str, ...] = _DEFAULT_ADDONS
    for words, addons in _ADDON_RULES:
        if any(word in name for word in words):
            chosen = addons if isinstance(addons, tuple) else (addons,)
            break
    return [a for a in chosen if a in ADDONS][:limit]


def ingredients_for(name: str) -> str | None:
    """Drafted ingredients for a dish, or None if we don't have them."""
    return INGREDIENTS.get(name.lower().strip())


def has_allergen_hint(ingredients: str) -> bool:
    return any(word in ingredients.lower() for word in _ALLERGEN_HINTS)


def food_categories() -> dict[str, list[str]]:
    """Category label -> dish names, for browsing."""
    out: dict[str, list[str]] = {}
    for name, (_price, category) in FOOD_MENU.items():
        out.setdefault(category, []).append(name)
    return out


def rupees(amount: float) -> str:
    """Speakable price. Whole rupees only; the catalogue has no paise."""
    if amount == 0:
        return "complimentary"
    return f"{int(amount)} rupees"


__all__: list[str] = [
    "FOOD_MENU",
    "ADDONS",
    "SERVICE_CATEGORIES",
    "INGREDIENTS",
    "find_food",
    "find_service",
    "find_addons",
    "addons_for",
    "ingredients_for",
    "has_allergen_hint",
    "food_categories",
    "rupees",
]
