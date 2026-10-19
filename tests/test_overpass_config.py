import json
import os
import unittest
from collections import OrderedDict
from unittest.mock import patch

import osmnx as ox
import requests
from osmnx import _overpass
from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError

from REQreate import overpass_config


def _response(url, status, body):
    response = requests.Response()
    response.status_code = status
    response.reason = {200: "OK", 403: "Forbidden"}.get(status, "")
    response.url = url
    response._content = body.encode()
    return response


class OverpassFailoverTests(unittest.TestCase):
    def setUp(self):
        self.mirrors = ["https://first.example/api/interpreter", "https://second.example/api/interpreter"]

    def _run_with_retries(self, request, attempts_per_mirror=1):
        set_url = patch.object(overpass_config, "_set_overpass_url").start()
        self.addCleanup(patch.stopall)
        patch.object(overpass_config, "get_mirrors", return_value=self.mirrors).start()
        patch.object(overpass_config, "ATTEMPTS_PER_MIRROR", attempts_per_mirror).start()
        patch.object(overpass_config.time, "sleep").start()
        result = overpass_config._with_failover(request)()
        return result, set_url

    def test_malformed_successful_response_fails_over(self):
        html_error = InsufficientResponseError("HTML instead of JSON")
        html_error.__cause__ = json.JSONDecodeError("invalid JSON", "<html>", 0)
        calls = []

        def request():
            calls.append(None)
            if len(calls) == 1:
                raise html_error
            return {"elements": []}

        result, set_url = self._run_with_retries(request)

        self.assertEqual(result, {"elements": []})
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            [call.args[0] for call in set_url.call_args_list],
            self.mirrors,
        )

    def test_response_status_error_fails_over(self):
        error = ResponseStatusCodeError("server rejected request")
        calls = []

        def request():
            calls.append(None)
            if len(calls) == 1:
                raise error
            return {"elements": []}

        result, set_url = self._run_with_retries(request)

        self.assertEqual(result, {"elements": []})
        self.assertEqual(len(calls), 2)
        self.assertEqual(set_url.call_count, 2)

    def test_empty_feature_result_does_not_retry(self):
        empty_result = InsufficientResponseError("No matching features.")
        calls = []

        def request():
            calls.append(None)
            raise empty_result

        with self.assertRaises(InsufficientResponseError) as raised:
            self._run_with_retries(request)

        self.assertIs(raised.exception, empty_result)
        self.assertEqual(len(calls), 1)

    def test_query_error_propagates_immediately(self):
        query_error = ValueError("invalid query")
        calls = []

        def request():
            calls.append(None)
            raise query_error

        with self.assertRaises(ValueError) as raised:
            self._run_with_retries(request)

        self.assertIs(raised.exception, query_error)
        self.assertEqual(len(calls), 1)

    def test_client_error_moves_to_next_mirror_without_retrying(self):
        forbidden = ResponseStatusCodeError("'first.example' responded: 403 Forbidden <html>")
        calls = []

        def request():
            calls.append(None)
            if len(calls) == 1:
                raise forbidden
            return {"elements": []}

        result, set_url = self._run_with_retries(request, attempts_per_mirror=3)

        self.assertEqual(result, {"elements": []})
        self.assertEqual(len(calls), 2)
        self.assertEqual([call.args[0] for call in set_url.call_args_list], self.mirrors)

    def test_rate_limit_retries_same_mirror(self):
        too_many = ResponseStatusCodeError("'first.example' responded: 429 Too Many Requests")
        calls = []

        def request():
            calls.append(None)
            if len(calls) == 1:
                raise too_many
            return {"elements": []}

        result, set_url = self._run_with_retries(request, attempts_per_mirror=2)

        self.assertEqual(result, {"elements": []})
        self.assertEqual([call.args[0] for call in set_url.call_args_list], [self.mirrors[0]])

    def test_transport_error_retries_same_mirror(self):
        calls = []

        def request():
            calls.append(None)
            if len(calls) == 1:
                raise ConnectionError("connection refused")
            return {"elements": []}

        result, set_url = self._run_with_retries(request, attempts_per_mirror=2)

        self.assertEqual(result, {"elements": []})
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            [call.args[0] for call in set_url.call_args_list],
            [self.mirrors[0]],
        )


class OverpassUrlTests(unittest.TestCase):
    """osmnx 2 appends "/interpreter" to settings.overpass_url itself."""

    def setUp(self):
        saved = (ox.settings.overpass_url, ox.settings.use_cache, ox.settings.overpass_rate_limit)

        def restore():
            ox.settings.overpass_url, ox.settings.use_cache, ox.settings.overpass_rate_limit = saved

        self.addCleanup(restore)
        ox.settings.use_cache = False
        ox.settings.overpass_rate_limit = False

    def test_default_mirrors_are_api_base_urls(self):
        for mirror in overpass_config.DEFAULT_OVERPASS_MIRRORS:
            self.assertFalse(mirror.rstrip("/").endswith("/interpreter"), mirror)

    def test_env_override_accepts_either_form(self):
        value = "https://a.example/api/interpreter, https://b.example/api/"
        with patch.dict(os.environ, {"REQREATE_OVERPASS_URL": value}):
            self.assertEqual(
                overpass_config.get_mirrors(), ["https://a.example/api", "https://b.example/api"]
            )

    def test_set_overpass_url_stores_api_base(self):
        overpass_config._set_overpass_url("https://a.example/api/interpreter")
        self.assertEqual(ox.settings.overpass_url, "https://a.example/api")

    def test_osmnx_posts_to_a_single_interpreter_path(self):
        """Run osmnx's real request code with only the network call replaced.

        The first mirror answers 403, as the public servers do for
        .../interpreter/interpreter; failover must move on after one attempt.
        """
        posted = []

        def post(url, **kwargs):
            posted.append(url)
            if len(posted) == 1:
                return _response(url, 403, "<html>403 Forbidden</html>")
            return _response(url, 200, json.dumps({"elements": []}))

        mirrors = ["https://first.example/api/interpreter", "https://second.example/api"]
        with patch.object(_overpass.requests, "post", side_effect=post), \
                patch.object(_overpass._http, "_config_dns"), \
                patch.object(overpass_config, "get_mirrors", return_value=mirrors), \
                patch.object(overpass_config.time, "sleep"):
            request = overpass_config._with_failover(_overpass._overpass_request)
            result = request(OrderedDict(data="[out:json];node(1);out;"))

        self.assertEqual(result, {"elements": []})
        self.assertEqual(
            posted,
            ["https://first.example/api/interpreter", "https://second.example/api/interpreter"],
        )


if __name__ == "__main__":
    unittest.main()
