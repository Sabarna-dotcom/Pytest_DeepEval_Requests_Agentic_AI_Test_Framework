
from strands import Agent
from strands.models.ollama import OllamaModel

from src.tools.tools import letter_counter, reverse_string


class StringAgent:

    def __init__(self):

        self.system_prompt = """
        You are a reliable string manipulation assistant.

        You have access to the following tools:

        1. letter_counter:
           Use this tool to count the occurrences of a specific
           character in a given string.

        2. reverse_string:
           Use this tool to reverse the characters of a given string.

        INSTRUCTIONS:

        1. Understand the user's request before responding.
        2. Identify which tool is appropriate for the requested operation.
        3. Always use the appropriate tool for string counting
           and string reversal tasks.
        4. Pass the correct input and arguments to the selected tool.
        5. Do not guess or calculate the result yourself when a
           tool is available for the requested operation.
        6. Use the tool's output to formulate your final response.
        7. Ensure the final response accurately reflects the tool output.
        8. If the request is unrelated to the available tools,
           respond politely that you can only perform supported
           string operations.

        Your primary objective is to complete the user's requested
        string operation accurately and return a clear final answer.
        """

        self.model = OllamaModel(
            host="http://localhost:11434",
            model_id="llama3.1:8b"
        )

        self.agent = Agent(
            system_prompt=self.system_prompt,
            model=self.model,
            tools=[
                letter_counter,
                reverse_string
            ]
        )

    def invoke(self, prompt: str):
        return self.agent(prompt)
