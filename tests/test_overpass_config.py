import json
import unittest
from unittest.mock import patch

from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError

from REQreate import overpass_config


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


if __name__ == "__main__":
    unittest.main()
