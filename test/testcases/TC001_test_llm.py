
import pytest
import time

from utilities.customLogger import LogGen
from utilities.JsonUtility import JsonUtility


# Load test data
data = JsonUtility.read_json("TC001_llm_testdata.json")

# Extract scenarios
test_data = data["tc_001"]


@pytest.mark.smoke
@pytest.mark.regression
class TestLLMEvaluation:

    @pytest.mark.parametrize(
        "scenario",
        test_data,
        ids=[
            f"Scenario_{index}"
            for index in range(1, len(test_data) + 1)
        ]
    )
    def test_evaluation_llm_response(self, judge_model, scenario):

        logger = LogGen.loggen()

        prompt = scenario["prompt"]

        logger.info("=" * 70)
        logger.info("STARTING LLM EVALUATION TEST")
        logger.info("Prompt: %s", prompt)

        try:
            start_time = time.perf_counter()

            response = judge_model.generate(prompt)

            execution_time = time.perf_counter() - start_time

            logger.info(
                "LLM inference completed in %.2f seconds",
                execution_time
            )

            assert response is not None, "LLM returned None"

            actual_response = (
                response[0] if isinstance(response, tuple) else response
            )

            assert isinstance(actual_response, str), (
                f"Unexpected response type: {type(response)}"
            )

            assert actual_response.strip(), (
                "LLM generated an empty response"
            )

            logger.info("Response validation successful")
            logger.info("Generated Response: %s", actual_response)
            logger.info(
                "Response Length: %d characters",
                len(actual_response)
            )

        except Exception:
            logger.exception("LLM EVALUATION TEST FAILED")
            raise

        finally:
            logger.info("LLM TEST EXECUTION FINISHED")
