"""REQreate - instance generator for demand responsive transit systems.

The generation entry point is :func:`REQreate.input_json.input_json`, which
reads a JSON configuration and writes an instance. The bundled Streamlit
interface is launched with the ``reqreate app`` command (see cli.py).
"""

__version__ = "0.1.2"

# Profiling is off unless REQREATE_PROFILE is set; when it is, wrap the osmnx
# calls so their cost is attributed rather than landing in "unaccounted".
from . import profiling as _profiling  # noqa: E402
_profiling.instrument_osmnx()
