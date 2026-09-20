"""
Overpass API resilience layer.

The public Overpass instance at overpass-api.de regularly refuses connections
(TCP ECONNREFUSED) or rate-limits callers, especially from shared-IP hosts such
as Hugging Face Spaces or Replit, where many users leave from the same address.
A single refusal used to abort a whole generation run after the user had already
waited several minutes.

This module:
  * configures osmnx's request settings under the correct names (osmnx renamed
    several of them in 2.0, so setting the old ones is a silent no-op);
  * installs a wrapper around osmnx's single Overpass chokepoint that retries
    with exponential backoff and then fails over to the next mirror.

Call configure_overpass() once before any OSM download. It is idempotent.

Set REQREATE_OVERPASS_URL to a comma-separated list of interpreter URLs to
override the mirror list, e.g. when running behind a private Overpass instance.
"""

import os
import time
import warnings

import osmnx as ox

# Tried in order. The first is osmnx's own default.
DEFAULT_OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]

# Per mirror, before moving on to the next one.
ATTEMPTS_PER_MIRROR = 3
BACKOFF_SECONDS = 5

_configured = False


class OverpassUnavailableError(RuntimeError):
    """Every Overpass mirror refused or failed to answer."""


def get_mirrors():
    """Mirror list, honouring the REQREATE_OVERPASS_URL override."""

    override = os.environ.get("REQREATE_OVERPASS_URL", "").strip()
    if override:
        mirrors = [url.strip() for url in override.split(",") if url.strip()]
        if mirrors:
            return mirrors
    return list(DEFAULT_OVERPASS_MIRRORS)


def _set_overpass_url(url):
    """Point osmnx at `url`, across both the 1.x and 2.x setting names."""

    if hasattr(ox.settings, "overpass_url"):
        # osmnx >= 1.6: full interpreter URL.
        ox.settings.overpass_url = url
    elif hasattr(ox.settings, "overpass_endpoint"):
        # osmnx < 1.6: endpoint without the trailing /interpreter.
        ox.settings.overpass_endpoint = url[: -len("/interpreter")] if url.endswith("/interpreter") else url
    else:
        warnings.warn("osmnx exposes no Overpass URL setting; mirror failover is disabled")


def _network_error(exc):
    """True when `exc` looks like a transport failure worth retrying elsewhere.

    Matched by name rather than by class so that the module does not have to
    import requests or urllib3 just to identify them.
    """

    names = {type(cause).__name__ for cause in _causes(exc)}
    return bool(names & {
        "ConnectionError",
        "NewConnectionError",
        "MaxRetryError",
        "ConnectTimeout",
        "ReadTimeout",
        "Timeout",
        "ConnectionRefusedError",
        "ChunkedEncodingError",
        "ProtocolError",
        "RemoteDisconnected",
        "SSLError",
    })


def _causes(exc):
    """The exception plus its __cause__/__context__ chain."""

    seen = []
    while exc is not None and not any(exc is s for s in seen):
        seen.append(exc)
        exc = exc.__cause__ or exc.__context__
    return seen


def _with_failover(original):
    """Wrap osmnx's Overpass request function with retry + mirror failover."""

    def wrapper(*args, **kwargs):
        mirrors = get_mirrors()
        last_error = None

        for mirror in mirrors:
            _set_overpass_url(mirror)

            for attempt in range(1, ATTEMPTS_PER_MIRROR + 1):
                try:
                    return original(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - re-raised below
                    if not _network_error(exc):
                        # A real query error (bad syntax, empty result, ...).
                        # Another mirror would answer the same way.
                        raise
                    last_error = exc
                    if attempt < ATTEMPTS_PER_MIRROR:
                        delay = BACKOFF_SECONDS * (2 ** (attempt - 1))
                        print(f"Overpass request to {mirror} failed ({exc}); retrying in {delay}s")
                        time.sleep(delay)
                    else:
                        print(f"Overpass mirror {mirror} unreachable; trying the next one")

        # Leave the setting on the default so a later run starts clean.
        _set_overpass_url(mirrors[0])
        raise OverpassUnavailableError(
            "Could not reach any Overpass API mirror, so the OpenStreetMap data "
            "for this instance could not be downloaded. This is an availability "
            "problem with the public Overpass servers (or with this host's "
            "outbound network), not a problem with the requested location.\n"
            f"Mirrors tried: {', '.join(mirrors)}\n"
            f"Last error: {last_error}\n"
            "Retry in a few minutes, or set REQREATE_OVERPASS_URL to an "
            "Overpass instance you can reach."
        ) from last_error

    wrapper.__wrapped__ = original
    return wrapper


def configure_overpass(timeout=1800):
    """Apply request settings and install the failover wrapper. Idempotent."""

    global _configured

    # requests_timeout (osmnx >= 2.0) was called timeout in 1.x. Setting a name
    # osmnx does not know is silently accepted and does nothing, so only set
    # names that actually exist.
    if hasattr(ox.settings, "requests_timeout"):
        ox.settings.requests_timeout = timeout
    elif hasattr(ox.settings, "timeout"):
        ox.settings.timeout = timeout

    # Respect the server's advertised rate limit rather than hammering it; this
    # matters because POI retrieval issues ten queries back to back.
    if hasattr(ox.settings, "overpass_rate_limit"):
        ox.settings.overpass_rate_limit = True

    # Cache responses so a failure partway through a run does not mean
    # re-downloading everything that already succeeded.
    if hasattr(ox.settings, "use_cache"):
        ox.settings.use_cache = True

    if _configured:
        return

    _set_overpass_url(get_mirrors()[0])

    # All of osmnx's Overpass traffic funnels through one private function, and
    # its callers reference it as a module global, so wrapping it here covers
    # every graph_from_* / features_from_* call in the codebase.
    module = getattr(ox, "_overpass", None) or getattr(ox, "downloader", None)
    request_func = getattr(module, "_overpass_request", None) if module else None

    if request_func is None:
        warnings.warn(
            "Could not find osmnx's Overpass request function; mirror failover "
            "is disabled. A refused connection will abort the run."
        )
    elif not hasattr(request_func, "__wrapped__"):
        module._overpass_request = _with_failover(request_func)

    _configured = True
