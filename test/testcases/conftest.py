import sys
from pathlib import Path

from deepeval.test_case import ToolCall
from strands.hooks import AfterToolCallEvent
from deepeval.tracing import observe, update_current_trace
import requests

from datetime import datetime

import pytest

from deepeval.models import OllamaModel

from utilities.customLogger import LogGen
from utilities.readProperties import ReadConfig
from deepeval.tracing import observe, update_current_trace



# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add project root to Python path
sys.path.insert(0, str(PROJECT_ROOT))

# Add test directory to Python path
TEST_ROOT = PROJECT_ROOT / "test"
sys.path.insert(0, str(TEST_ROOT))

from src.agents.string_agent import StringAgent

@pytest.fixture(scope="session")
def judge_model():

    logger = LogGen.loggen()

    model_name = ReadConfig.getEvaluationModel()

    logger.info("=" * 70)
    logger.info("INITIALIZING OLLAMA JUDGE MODEL")
    logger.info("Evaluation Model: %s", model_name)
    logger.info("Ollama Base URL: %s", ReadConfig.getLLMUrl())
    logger.info("Temperature: %s", ReadConfig.getTemperature())

    try:
        model = OllamaModel(
            model=model_name,
            base_url=ReadConfig.getLLMUrl(),
            temperature=ReadConfig.getTemperature()
        )

        logger.info("Ollama Judge Model initialized successfully")

        yield model

        logger.info("Ollama Judge Model fixture execution completed")
        logger.info("=" * 70)

    except Exception:
        logger.exception("Failed to initialize Ollama Judge Model")
        raise


@pytest.fixture(scope="session")
def string_agent():

    agent = StringAgent()

    yield agent



@pytest.fixture(scope="session")
def api_base_url():
    return "http://127.0.0.1:8000"


@pytest.fixture(scope="session")
def api_client():
    session = requests.Session()

    yield session

    session.close()


@pytest.fixture(scope="session")
def traced_agent(string_agent):

    tools_called = []

    def capture_tool_call(event: AfterToolCallEvent):
        tools_called.append(event.tool_use["name"])

    # Register only once
    string_agent.agent.hooks.add_callback(
        AfterToolCallEvent,
        capture_tool_call
    )

    @observe(
        type="agent",
        available_tools=["letter_counter", "reverse_string"]
    )
    def execute_agent(prompt):

        # Reset captured calls for this execution
        tools_called.clear()

        response = string_agent.invoke(prompt)

        update_current_trace(
            input=prompt,
            output=str(response)
        )

        return {
            "response": response,
            "tools_called": list(tools_called)
        }

    return execute_agent


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):

    # Create reports directory
    report_dir = Path(__file__).resolve().parent.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp-based HTML report name
    report_file = (
        report_dir /
        f"DeepEval_Report_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.html"
    )

    config.option.htmlpath = str(report_file)
    config.option.self_contained_html = True


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):

    # Capture test session start time
    session.start_time = datetime.now()


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):

    # Capture test session end time
    session.end_time = datetime.now()

    # Calculate total execution duration
    session.duration = (
        session.end_time - session.start_time
    ).total_seconds()


def pytest_html_results_summary(prefix, summary, postfix, session):

    # Add execution details to HTML report
    start_time = session.start_time
    end_time = session.end_time
    duration = session.duration

    prefix.extend([
        f"<h3>Execution Details</h3>",
        f"<p><b>Test Start Time:</b> "
        f"{start_time.strftime('%d-%m-%Y %H:%M:%S')}</p>",

        f"<p><b>Test End Time:</b> "
        f"{end_time.strftime('%d-%m-%Y %H:%M:%S')}</p>",

        f"<p><b>Total Execution Time:</b> "
        f"{duration:.2f} seconds</p>"
    ])
