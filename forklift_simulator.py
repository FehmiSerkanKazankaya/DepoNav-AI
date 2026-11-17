"""Self-contained, no-network HTML/SVG forklift picking animation."""
import json
from pathlib import Path

TEMPLATE=Path(__file__).resolve().parent/'assets'/'forklift_simulator.html'


def render_simulator(route_details:dict, storage_background:list[list[float]])->str:
    data={
        'io':route_details['io'],
        'legs':route_details['legs'],
        'racks':route_details['racks'],
        'total_stops':route_details['total_stops'],
        'total_units':route_details['total_units'],
        'distance':route_details['distance'],
        'geometry':route_details['geometry'],
        'background':storage_background,
    }
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    payload=payload.replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    return TEMPLATE.read_text(encoding='utf-8').replace('__ROUTE_JSON__',payload)
