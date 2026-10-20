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

Set REQREATE_OVERPASS_URL to a comma-separated list of Overpass API URLs to
override the mirror list, e.g. when running behind a private Overpass instance.
Either form is accepted: https://host/api or https://host/api/interpreter.
"""

import os
import re
import time
import warnings

import osmnx as ox

# Tried in order. The first is osmnx's own default. These are API base URLs:
# osmnx 2 appends "/interpreter" itself, so they must not end in it.
#
# These must be planet-wide instances. A regional mirror (there are several,
# covering a single country) would answer a query for a place it does not hold
# with an empty result rather than an error, which would silently produce an
# instance with no network in it -- worse than the failure it was meant to
# work around. Check coverage before adding a mirror here.
#
# overpass.kumi.systems and overpass.private.coffee used to be listed, but in
# October 2026 both timed out on every query from a GitHub runner, so failing
# over to them only added waiting before the same error.
DEFAULT_OVERPASS_MIRRORS = [
    "https://overpass-api.de/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
]

# Per mirror, before moving on to the next one.
ATTEMPTS_PER_MIRROR = 3
BACKOFF_SECONDS = 5

_configured = False


class OverpassUnavailableError(RuntimeError):
    """Every Overpass mirror refused or failed to answer."""


def _api_base(url):
    """`url` without a trailing "/interpreter", the form osmnx 2 expects."""

    url = url.strip().rstrip("/")
    if url.endswith("/interpreter"):
        url = url[: -len("/interpreter")]
    return url


def get_mirrors():
    """Mirror API base URLs, honouring the REQREATE_OVERPASS_URL override."""

    override = os.environ.get("REQREATE_OVERPASS_URL", "").strip()
    if override:
        mirrors = [_api_base(url) for url in override.split(",") if url.strip()]
        if mirrors:
            return mirrors
    return list(DEFAULT_OVERPASS_MIRRORS)


def get_cache_folder():
    """Shared OpenStreetMap download cache, honouring REQREATE_CACHE_DIR.

    Defaults to %LOCALAPPDATA%\\reqreate\\osm-cache on Windows and
    ~/.cache/reqreate/osm-cache elsewhere.
    """

    override = os.environ.get("REQREATE_CACHE_DIR", "").strip()
    if override:
        return override
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "reqreate", "osm-cache")


def _set_overpass_url(url):
    """Point osmnx at the Overpass API at `url`.

    Both osmnx 2's overpass_url and 1.x's overpass_endpoint hold the API base
    and osmnx appends "/interpreter" itself. Passing the full interpreter URL
    sends every query to .../api/interpreter/interpreter, which the public
    servers answer with 403 Forbidden.
    """

    url = _api_base(url)
    if hasattr(ox.settings, "overpass_url"):
        ox.settings.overpass_url = url
    elif hasattr(ox.settings, "overpass_endpoint"):
        ox.settings.overpass_endpoint = url
    else:
        warnings.warn("osmnx exposes no Overpass URL setting; mirror failover is disabled")


def _network_error(exc):
    """True when `exc` looks like a failure worth retrying on another mirror.

    Matched by name rather than by class so that the module does not have to
    import requests or urllib3 just to identify transport errors. OSMnx wraps
    malformed successful JSON responses in InsufficientResponseError with the
    JSONDecodeError as its cause; empty feature results have no such cause.
    """

    causes = _causes(exc)
    names = {type(cause).__name__ for cause in causes}
    transport_error = bool(names & {
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
    server_error = "ResponseStatusCodeError" in names
    malformed_successful_response = (
        "InsufficientResponseError" in names
        and "JSONDecodeError" in {type(cause).__name__ for cause in causes}
    )
    return transport_error or server_error or malformed_successful_response


def _status_code(exc):
    """HTTP status of an osmnx ResponseStatusCodeError in `exc`'s chain, or None.

    osmnx only puts the status in the message: "'host' responded: 403 ...".
    """

    for cause in _causes(exc):
        if type(cause).__name__ == "ResponseStatusCodeError":
            match = re.search(r"responded: (\d{3})", str(cause))
            if match:
                return int(match.group(1))
    return None


def _retry_same_mirror(exc):
    """False when this mirror will answer the same way however often we ask.

    A 4xx other than 429 is the server refusing the request itself (403 for a
    blocked client or a bad path, for instance). Repeating it only adds to the
    reasons for blocking us, so go straight to the next mirror instead.
    """

    status = _status_code(exc)
    return not (status is not None and 400 <= status < 500 and status != 429)


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
                    if attempt < ATTEMPTS_PER_MIRROR and _retry_same_mirror(exc):
                        delay = BACKOFF_SECONDS * (2 ** (attempt - 1))
                        print(f"Overpass request to {mirror} failed ({exc}); retrying in {delay}s")
                        time.sleep(delay)
                    else:
                        print(f"Overpass mirror {mirror} unreachable; trying the next one")
                        break

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
    # One cache per user rather than ./cache in whatever folder the command
    # runs from, so every run reuses earlier downloads of the same city.
    if hasattr(ox.settings, "cache_folder"):
        ox.settings.cache_folder = get_cache_folder()

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
