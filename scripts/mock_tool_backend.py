"""Mock of the voice-ai-agent backend's tool routes (contract section 5).

Lets the fork's tool-calling loop be built and tested before Repo A exists.
On integration day, point TOOL_API_BASE_URL at the real backend instead; if
both sides honoured the contract, nothing else changes.

    python scripts/mock_tool_backend.py            # port 8081
    TOOL_API_KEY=secret python scripts/mock_tool_backend.py

Force the honest-failure path (contract section 5 asks for one failure case):

    MOCK_FAIL=book_taxi python scripts/mock_tool_backend.py

This mock deliberately implements the CONTRACT, not the business logic: fixed
canned totals, no catalog, no real bookings. It is not a stand-in for Repo A.
"""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.environ.get("MOCK_PORT", "8081"))
API_KEY = os.environ.get("TOOL_API_KEY")
# Comma-separated tool names that should return their ok:false path.
FAIL_TOOLS = {t for t in os.environ.get("MOCK_FAIL", "").split(",") if t}

SUCCESS = {
    "get_guest_by_room": lambda a: (
        "I have you as Priya Sharma in room {}.".format(a.get("room_number", "?")),
        {
            "found": True,
            "guest_name": "Priya Sharma",
            "guest_phone": "+91-99999-00000",
            "guest_email": "priya@example.com",
        },
    ),
    "book_taxi": lambda a: (
        "Your taxi is booked. Raju will pick you up at {}, vehicle DL-1234.".format(
            a.get("pickup_time", "the requested time")
        ),
        {"booking_id": "MOCK1", "driver": "Raju", "vehicle": "DL-1234"},
    ),
    "create_food_order": lambda a: (
        "That's ordered - two burgers and a pizza, 640 rupees, "
        "about 30 minutes to your room.",
        {"order_id": "F77", "total": 640.0},
    ),
    "create_laundry_request": lambda a: (
        "Your laundry pickup is arranged for {}.".format(
            a.get("pickup_time", "later today")
        ),
        {"request_id": "L21", "total": 150.0},
    ),
    "create_maintenance_ticket": lambda a: (
        "I've logged that with maintenance; someone will be up shortly.",
        {"ticket_id": "M09"},
    ),
    "create_bill_and_payment_link": lambda a: (
        "Your bill comes to 320 rupees. I've sent a payment link to your phone.",
        {
            "order_id": "B12",
            "payment_link": "https://pay.example.com/mock",
            "total": 320.0,
        },
    ),
}

FAILURE = {
    "book_taxi": (
        "I'm sorry, no drivers are free right now. "
        "Shall I have the front desk call you?",
        "NO_DRIVERS_AVAILABLE",
    ),
    "get_guest_by_room": (
        "I'm not finding a guest registered to that room. "
        "Could you tell me the room number again?",
        "GUEST_NOT_FOUND",
    ),
    "create_food_order": (
        "I couldn't find those on tonight's menu. Shall I read you what we have?",
        "ITEM_NOT_IN_CATALOG",
    ),
}
DEFAULT_FAILURE = (
    "I'm sorry, I couldn't complete that just now. "
    "Shall I have the front desk follow up?",
    "MOCK_FAILURE",
)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith("/internal/tools/"):
            self._send(404, {"detail": "not found"})
            return

        name = self.path.rsplit("/", 1)[-1]

        # Contract section 1: auth is enforced even on localhost.
        if API_KEY and self.headers.get("X-Tool-Api-Key") != API_KEY:
            self._send(401, {"detail": "bad or missing X-Tool-Api-Key"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            args = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"detail": "bad json"})
            return

        print(f"  -> {name} {args}")

        if name not in SUCCESS:
            self._send(404, {"detail": f"unknown tool {name}"})
            return

        if name in FAIL_TOOLS:
            speakable, code = FAILURE.get(name, DEFAULT_FAILURE)
            self._send(
                200,
                {"ok": False, "speakable": speakable, "error_code": code, "data": {}},
            )
            return

        speakable, data = SUCCESS[name](args)
        self._send(200, {"ok": True, "speakable": speakable, "data": data})

    def log_message(self, *a) -> None:
        pass  # POSTs are logged above; suppress the default noise.


if __name__ == "__main__":
    print(f"Mock tool backend on http://localhost:{PORT}")
    print(f"  auth     : {'X-Tool-Api-Key required' if API_KEY else 'disabled'}")
    print(f"  failing  : {', '.join(sorted(FAIL_TOOLS)) or '(none)'}")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
