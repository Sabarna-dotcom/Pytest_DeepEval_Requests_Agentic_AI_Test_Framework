
import pytest

from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from utilities.JsonUtility import JsonUtility
from utilities.customLogger import LogGen
from utilities.readProperties import ReadConfig


logger = LogGen.loggen()

# Load test data
data = JsonUtility.read_json("TC003_tool_correctness.json")
test_data = data["tc_003"]


@pytest.mark.llm
@pytest.mark.regression
class TestToolCorrectness:

    @pytest.mark.parametrize(
        "scenario",
        test_data,
        ids=[
            f"Scenario_{index}"
            for index in range(1, len(test_data) + 1)
        ]
    )
    def test_agent_tool_correctness(
        self,
        traced_agent,
        judge_model,
        scenario
    ):

        prompt = scenario["prompt"]

        logger.info("=" * 60)
        logger.info("Starting Tool Correctness Test")
        logger.info("Test Scenario: %s", scenario["task"])
        logger.info("Prompt: %s", prompt)

        try:
            # Step 1: Execute agent and capture actual tool calls
            result = traced_agent(prompt)

            response = result["response"]
            actual_tools = result["tools_called"]

            logger.info("Agent Response: %s", response)
            logger.info("Actual tools called: %s", actual_tools)

            # Step 2: Get expected tools from test data
            expected_tool_names = scenario["expected_tools"]

            logger.info("Expected tools: %s", expected_tool_names)

            # Step 3: Deterministic validation of tool calls
            assert actual_tools == expected_tool_names, (
                f"Tool mismatch! "
                f"Expected: {expected_tool_names}, "
                f"Actual: {actual_tools}"
            )

            logger.info("Deterministic Tool Validation: PASSED")

            # Step 4: Convert actual tool names into ToolCall objects
            tools_called = [
                ToolCall(name=tool_name)
                for tool_name in actual_tools
            ]

            # Step 5: Convert expected tool names into ToolCall objects
            expected_tools = [
                ToolCall(name=tool_name)
                for tool_name in expected_tool_names
            ]

            # Step 6: Define tools available to the agent
            available_tools = [
                ToolCall(name="reverse_string"),
                ToolCall(name="letter_counter")
            ]

            # Step 7: Create DeepEval test case
            test_case = LLMTestCase(
                input=prompt,
                actual_output=str(response),
                tools_called=tools_called,
                expected_tools=expected_tools
            )

            # Step 8: Configure Tool Correctness Metric
            tool_correctness = ToolCorrectnessMetric(
                threshold=ReadConfig.getThreshold(),
                model=judge_model,
                available_tools=available_tools,
                include_reason=True,
                verbose_mode=True
            )

            # Step 9: Execute DeepEval metric validation
            assert_test(
                test_case=test_case,
                metrics=[tool_correctness]
            )

            logger.info("DeepEval Tool Correctness Validation: PASSED")

        finally:
            logger.info("Tool Correctness Test Execution Completed")
            logger.info("=" * 60)
