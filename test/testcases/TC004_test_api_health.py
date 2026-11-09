
import pytest

from utilities.customLogger import LogGen

logger = LogGen.loggen()


@pytest.mark.api
@pytest.mark.smoke
class TestAPIHealth:

    def test_health_check(
        self,
        api_client,
        api_base_url
    ):

        logger.info("Starting API Health Check")

        response = api_client.get(
            f"{api_base_url}/health",
            timeout=10
        )

        logger.info("Status Code: %s", response.status_code)
        logger.info("Response: %s", response.text)

        assert response.status_code == 200

        response_data = response.json()

        assert response_data["status"] == "UP"
        assert response_data["message"] == (
            "String Agent API is running"
        )

        logger.info("API Health Check: PASSED")
