"""
Carrier integration layer.

`CarrierClient` is the interface the rest of the app depends on. In this demo it is backed by
`SimulatedCarrierClient`, a deterministic shipment simulator: the same tracking number always
produces the same shipment (seeded by a hash of the number), with a realistic scan-event
history and a dwell time computed from the last scan. In production a real carrier or
tracking-aggregator client would implement the same interface without touching the AI layer.
"""
import hashlib
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Protocol

from app.ml_delay_model import CARRIERS, HUBS

TRACKING_PATTERN = re.compile(r"^(?=.*\d)[A-Z0-9][A-Z0-9-]{5,19}$")


@dataclass(frozen=True)
class Lane:
    origin: str
    origin_country: str
    destination: str
    destination_country: str
    route: tuple            # sorting hubs in order
    carriers: tuple
    transit_days: int
    customs: bool = False   # crosses an EU customs border

    @property
    def is_cross_border(self) -> bool:
        return self.origin_country != self.destination_country


LANES = [
    Lane("Berlin Warehouse", "DE", "Munich", "DE", ("Berlin South Center", "Leipzig Hub"),
         ("DHL Express", "Hermes Germany", "DPD Europe", "GLS Logistics"), 2),
    Lane("Frankfurt Hub", "DE", "Berlin", "DE", ("Frankfurt Gateway", "Leipzig Hub", "Potsdam Sorting Facility"),
         ("DHL Express", "Hermes Germany", "GLS Logistics"), 2),
    Lane("Berlin Warehouse", "DE", "Amsterdam", "NL", ("Berlin South Center", "Amsterdam Parcel Center"),
         ("PostNL", "DHL Express", "DPD Europe"), 3),
    Lane("Leipzig Warehouse", "DE", "Paris", "FR", ("Leipzig Hub", "Frankfurt Gateway"),
         ("Colissimo", "DPD Europe", "DHL Express"), 3),
    Lane("Zurich Warehouse", "CH", "Paris", "FR", ("Frankfurt Gateway", "Roissy CDG Airport Customs"),
         ("Colissimo", "DHL Express"), 4, customs=True),
]
assert all(c in CARRIERS for lane in LANES for c in lane.carriers)
assert all(h in HUBS for lane in LANES for h in lane.route)

CUSTOMS_REASONS = [
    "Missing Commercial Invoice",
    "Incorrect HS code on invoice",
    "Recipient EORI number missing",
    "Declared value does not match invoice",
]
RECIPIENT_NAMES = ["J. Smith", "A. Weber", "M. Rossi", "L. Martin", "S. de Jong", "K. Nowak"]

# Outcome mix for simulated shipments: (outcome, weight)
OUTCOMES = [("delivered", 35), ("on_time", 30), ("stalled", 20), ("customs", 10), ("not_found", 5)]


class CarrierClient(Protocol):
    def lookup(self, tracking_number: str) -> dict: ...


def _fmt(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M")


class SimulatedCarrierClient:
    """Deterministic shipment simulator. Timestamps are relative to `now`, so dwell times stay live."""

    # Fixed demo scenarios, built with the same machinery as simulated shipments
    PRESETS: Dict[str, dict] = {
        "SHIP-1001": {"lane": 1, "carrier": "DHL Express", "outcome": "stalled", "dwell": 70.0},
        "SHIP-2002": {"lane": 4, "carrier": "Colissimo", "outcome": "customs", "dwell": 44.0,
                      "exception_reason": "Missing Commercial Invoice"},
        "SHIP-3003": {"lane": 2, "carrier": "PostNL", "outcome": "delivered", "delivered_ago": 26.0,
                      "signed_by": "J. Smith"},
    }

    def __init__(self, now_fn: Callable[[], datetime] = datetime.now):
        self.now_fn = now_fn

    def lookup(self, tracking_number: str) -> dict:
        code = tracking_number.strip().upper()
        if not TRACKING_PATTERN.match(code):
            return self._not_found(code, "This is not a valid tracking number format.")

        rng = random.Random(int(hashlib.sha256(code.encode()).hexdigest(), 16))
        spec = self.PRESETS.get(code) or self._random_spec(rng)
        if spec["outcome"] == "not_found":
            return self._not_found(code, "No carrier has a record of this tracking number. It may be mistyped or not yet registered.")
        return self._build(code, spec, rng)

    def _random_spec(self, rng: random.Random) -> dict:
        outcome = rng.choices([o for o, _ in OUTCOMES], weights=[w for _, w in OUTCOMES])[0]
        lane_idx = next(i for i, l in enumerate(LANES) if l.customs) if outcome == "customs" else rng.randrange(len(LANES))
        lane = LANES[lane_idx]
        return {
            "lane": lane_idx,
            "carrier": rng.choice(lane.carriers),
            "outcome": outcome,
            "dwell": {"on_time": rng.uniform(2, 20), "stalled": rng.uniform(38, 90), "customs": rng.uniform(20, 70)}.get(outcome),
            "exception_reason": rng.choice(CUSTOMS_REASONS),
            "delivered_ago": rng.uniform(2, 96),
            "signed_by": rng.choice(RECIPIENT_NAMES),
            "hubs_reached": rng.randint(1, len(lane.route)),
        }

    @staticmethod
    def _not_found(code: str, note: str) -> dict:
        return {"tracking_number": code, "status": "NOT_FOUND", "carrier": "N/A", "notes": note}

    def _build(self, code: str, spec: dict, rng: random.Random) -> dict:
        lane = LANES[spec["lane"]]
        outcome = spec["outcome"]
        now = self.now_fn()

        # Delivered / customs parcels have passed every hub; in-transit ones may be part-way
        reached = spec.get("hubs_reached", len(lane.route))
        hubs = list(lane.route) if outcome in ("delivered", "customs") else list(lane.route[:reached])

        # Chronological event labels (timestamps added below)
        events = [(lane.origin, "Parcel picked up by carrier")]
        events += [(hub, "Arrived at sorting hub") for hub in hubs]
        if outcome == "customs":
            events[-1] = (hubs[-1], f"Held by customs: {spec['exception_reason']}")
        if outcome == "delivered":
            events += [(lane.destination, "Out for delivery"), (lane.destination, f"Delivered - signed by {spec['signed_by']}")]

        # Walk backwards from the last scan, spacing earlier scans 6-18h apart
        last_scan_ago = spec["delivered_ago"] if outcome == "delivered" else spec["dwell"]
        times = [now - timedelta(hours=last_scan_ago)]
        for _ in events[:-1]:
            times.insert(0, times[0] - timedelta(hours=rng.uniform(6, 18)))
        scan_events = [{"timestamp": _fmt(t), "location": loc, "event": ev} for t, (loc, ev) in zip(times, events)]

        record = {
            "tracking_number": code,
            "carrier": spec["carrier"],
            "service": "Cross-Border Express" if lane.is_cross_border else "Standard Parcel",
            "status": {"delivered": "DELIVERED", "customs": "CUSTOMS_HOLD"}.get(outcome, "IN_TRANSIT"),
            "origin": f"{lane.origin}, {lane.origin_country}",
            "destination": f"{lane.destination}, {lane.destination_country}",
            "last_hub": hubs[-1],
            "is_cross_border": int(lane.is_cross_border),
            "exception_flag": outcome == "customs",
            "scan_events": scan_events,
        }
        if outcome == "delivered":
            record["delivered_at"] = scan_events[-1]["timestamp"]
            record["signed_by"] = spec["signed_by"]
            record["dwell_time_hours"] = None
        else:
            record["dwell_time_hours"] = round(last_scan_ago, 1)
            record["eta"] = "Pending Clearance" if outcome == "customs" else (times[0] + timedelta(days=lane.transit_days)).strftime("%Y-%m-%d")
        if outcome == "customs":
            record["exception_reason"] = spec["exception_reason"]
        return record


carrier_client: CarrierClient = SimulatedCarrierClient()


def lookup_shipment(tracking_number: Optional[str]) -> Optional[Dict]:
    """Look up shipment in the partner carrier network."""
    if not tracking_number:
        return {
            "status": "NO_TRACKING_PROVIDED",
            "carrier": "N/A",
            "notes": "Customer did not supply a valid tracking code."
        }
    return carrier_client.lookup(tracking_number)
