
import pytest

from utilities.JsonUtility import JsonUtility
from utilities.customLogger import LogGen

logger = LogGen.loggen()

data = JsonUtility.read_json("TC006_api_functionality.json")
test_data = data["tc_006"]


@pytest.mark.api
@pytest.mark.regression
class TestAPIFunctionality:

    @pytest.mark.parametrize(
        "scenario",
        test_data,
        ids=[scenario["task"] for scenario in test_data]
    )
    def test_agent_functionality(
        self,
        api_client,
        api_base_url,
        scenario
    ):

        logger.info("=" * 60)
        logger.info("Starting API Functionality Test")
        logger.info("Scenario: %s", scenario["task"])

        response = api_client.post(
            f"{api_base_url}/agent/invoke",
            json=scenario["payload"],
            timeout=120
        )

        logger.info("Status Code: %s", response.status_code)
        logger.info("Response: %s", response.text)

        # Validate HTTP status
        assert response.status_code == 200

        # Validate response structure
        response_data = response.json()

        assert "response" in response_data
        assert isinstance(response_data["response"], str)

        # Validate actual agent output
        actual_output = response_data["response"]

        expected_output = scenario["expected_output"]

        assert expected_output.casefold() in actual_output.casefold(), (
            f"Expected output '{expected_output}' "
            f"not found in actual output '{actual_output}'"
        )

        logger.info("API Functionality Test: PASSED")
        logger.info("=" * 60)
