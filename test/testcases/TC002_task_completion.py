import pytest

from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.metrics import TaskCompletionMetric

from utilities.customLogger import LogGen
from utilities.JsonUtility import JsonUtility
from utilities.readProperties import ReadConfig


# Load test data
data = JsonUtility.read_json("TC002_task_completion.json")
test_data = data["tc_002"]


@pytest.mark.regression
class TestTaskCompletion:

    @pytest.mark.parametrize(
        "scenario",
        test_data,
        ids=[
            f"Scenario_{index}"
            for index in range(1, len(test_data) + 1)
        ]
    )
    def test_agent_task_completion(
        self, traced_agent, judge_model, scenario
    ):

        logger = LogGen.loggen()

        prompt = scenario["prompt"]
        task = scenario["task"]

        logger.info("=" * 70)
        logger.info("STARTING TASK COMPLETION TEST")
        logger.info("Task: %s", task)
        logger.info("Prompt: %s", prompt)

        # Configure DeepEval metric
        task_completion = TaskCompletionMetric(
            threshold=ReadConfig.getThreshold(),
            model=judge_model,
            task=task,
            include_reason=True,
            verbose_mode=True
        )

        # Create Golden
        golden = Golden(input=prompt)

        try:
            # Execute traced agent
            response = traced_agent(prompt)

            logger.info("Agent Response: %s", response)

            assert response is not None, "Agent returned None"

            # Evaluate complete agent execution trace
            assert_test(
                golden=golden,
                metrics=[task_completion]
            )

            logger.info("TASK COMPLETION EVALUATION PASSED")

        except Exception:
            logger.exception("TASK COMPLETION TEST FAILED")
            raise

        finally:
            logger.info("TASK COMPLETION TEST FINISHED")