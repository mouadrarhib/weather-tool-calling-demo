import unittest
from lesson import dispatch_tool_call, handle_tool_request
from unittest.mock import patch


class TestToolDispatcher(unittest.TestCase):
    def test_missing_arguments(self):
        request = {"name": "get_weather"}

        result = dispatch_tool_call(request)

        self.assertEqual(result["code"], "INVALID_ARGUMENTS")

    def test_unknown_tool(self):
        request = {
            "name": "send_email",
            "arguments": {}
        }

        result = dispatch_tool_call(request)

        self.assertEqual(result["code"], "UNKNOWN_TOOL")

    def test_blank_city(self):
        request = {
            "name" : "get_weather",
            "arguments" : {
                "city" : " "
            }
        }

        with patch("lesson.get_weather") as mock_weather:
            result = dispatch_tool_call(request)

            self.assertEqual(result["code"], "INVALID_CITY")
            mock_weather.assert_not_called()

    def test_valid_city_is_cleaned(self):
        request = {
            "name": "get_weather",
            "arguments": {"city": " Tokyo "}
        }
        sample_weather = {"city": "Tokyo", "temp_c": 18}

        with patch("lesson.get_weather", return_value=sample_weather) as mock_weather:
            result = dispatch_tool_call(request)

            mock_weather.assert_called_once_with(city="Tokyo")
            self.assertEqual(result, sample_weather)

    def test_weather_connection_failure(self):
        request = {
            "name": "get_weather",
            "arguments": {"city": "Tokyo"}
        }

        with patch("lesson.get_weather", side_effect=ConnectionError("Unavailable")):
            result = dispatch_tool_call(request)

            self.assertEqual(result["code"], "WEATHER_UNAVAILABLE")

    def test_malformed_argument_json(self):
        result = handle_tool_request("get_weather", '{"city":')

        self.assertEqual(result["code"], "INVALID_JSON")


    def test_argument_list_is_rejected(self):
        result = handle_tool_request("get_weather", "[]")

        self.assertEqual(result["code"], "INVALID_ARGUMENTS")


if __name__ == "__main__":
    unittest.main()