
import pytest

from utilities.JsonUtility import JsonUtility
from utilities.customLogger import LogGen

logger = LogGen.loggen()

data = JsonUtility.read_json("TC005_api_validation.json")
test_data = data["tc_005"]


@pytest.mark.api
@pytest.mark.regression
class TestAPIValidation:

    @pytest.mark.parametrize(
        "scenario",
        test_data,
        ids=[scenario["task"] for scenario in test_data]
    )
    def test_invalid_agent_request(
        self,
        api_client,
        api_base_url,
        scenario
    ):

        logger.info("Starting API Validation Test")
        logger.info("Scenario: %s", scenario["task"])

        response = api_client.post(
            f"{api_base_url}/agent/invoke",
            json=scenario["payload"],
            timeout=10
        )

        logger.info("Status Code: %s", response.status_code)
        logger.info("Response: %s", response.text)

        assert response.status_code == scenario["expected_status"]

        logger.info("API Validation Test: PASSED")
