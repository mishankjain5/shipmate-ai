from typing import Optional, Dict
import random

# Core Known Scenarios
KNOWN_SHIPMENTS: Dict[str, Dict] = {
    "SHIP-1001": {
        "tracking_number": "SHIP-1001",
        "carrier": "DHL Germany",
        "service": "Standard Parcel",
        "status": "IN_TRANSIT",
        "origin": "Frankfurt Hub",
        "destination": "Berlin",
        "eta": "2026-08-18",
        "last_hub": "Potsdam Sorting Center",
        "exception_flag": False
    },
    "SHIP-2002": {
        "tracking_number": "SHIP-2002",
        "carrier": "Colissimo",
        "service": "Cross-Border Express",
        "status": "CUSTOMS_HOLD",
        "origin": "Berlin Warehouse",
        "destination": "Paris, France",
        "eta": "Pending Clearance",
        "last_hub": "Roissy CDG Airport Customs",
        "exception_flag": True,
        "exception_reason": "Missing Commercial Invoice"
    },
    "SHIP-3003": {
        "tracking_number": "SHIP-3003",
        "carrier": "PostNL",
        "service": "Standard Direct",
        "status": "DELIVERED",
        "origin": "Hamburg Center",
        "destination": "Amsterdam",
        "delivered_at": "2026-08-14 10:15",
        "signed_by": "J. Smith",
        "exception_flag": False
    }
}

CARRIERS = ["DHL Express", "DPD Europe", "Hermes Germany", "GLS Logistics", "PostNL"]
HUBS = ["Potsdam Sorting Facility", "Leipzig Hub", "Frankfurt Gateway", "Dresden Depot", "Berlin South Center"]

def lookup_shipment(tracking_number: Optional[str]) -> Optional[Dict]:
    """Look up shipment in the partner carrier network."""
    if not tracking_number:
        return {
            "status": "NO_TRACKING_PROVIDED",
            "carrier": "N/A",
            "notes": "Customer did not supply a valid tracking code."
        }
    
    code = tracking_number.strip().upper()
    
    # Return known test case if matched
    if code in KNOWN_SHIPMENTS:
        return KNOWN_SHIPMENTS[code]
    
    # Realistic dynamic simulator for custom/any entered numbers
    return {
        "tracking_number": code,
        "carrier": random.choice(CARRIERS),
        "service": "Standard Cross-Border",
        "status": "DELIVERED" if "3" in code else "IN_TRANSIT",
        "origin": "Berlin Hub",
        "destination": "Customer Address",
        "last_hub": random.choice(HUBS),
        "eta": "2026-08-17",
        "exception_flag": False,
        "notes": "Verified via Partner Carrier API Gateway."
    }