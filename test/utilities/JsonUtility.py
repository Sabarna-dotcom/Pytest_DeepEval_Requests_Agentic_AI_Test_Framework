
import json
from pathlib import Path


class JsonUtility:

    @staticmethod
    def read_json(file_name):
        """
        Read any JSON file and return its contents.
        Supports different JSON structures.
        """

        file_path = (
            Path(__file__).resolve().parent.parent
            / "testData"
            / file_name
        )

        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def get_value(data, key):
        """
        Retrieve a value using a dictionary key.
        """

        return data.get(key)

    @staticmethod
    def get_nested_value(data, *keys):
        """
        Retrieve nested JSON values.
        """

        for key in keys:
            if not isinstance(data, dict):
                raise TypeError(f"Expected dictionary while accessing {key}")

            data = data[key]

        return data
