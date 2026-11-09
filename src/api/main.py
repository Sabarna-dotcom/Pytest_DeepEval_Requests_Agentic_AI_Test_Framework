
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agents.string_agent import StringAgent


# --------------------------------------------------
# 1. Initialize FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="String Agent API",
    description="REST API for invoking the Strands StringAgent",
    version="1.0.0"
)


# --------------------------------------------------
# 2. Initialize StringAgent
# --------------------------------------------------

string_agent = StringAgent()


# --------------------------------------------------
# 3. Define request schema
# --------------------------------------------------

class AgentRequest(BaseModel):

    prompt: str = Field(
        ...,
        min_length=1,
        description="Prompt to be processed by the StringAgent",
        examples=["Reverse the word Strands"]
    )


# --------------------------------------------------
# 4. Define response schema
# --------------------------------------------------

class AgentResponse(BaseModel):

    response: str


# --------------------------------------------------
# 5. Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "UP",
        "message": "String Agent API is running"
    }


# --------------------------------------------------
# 6. Agent invocation endpoint
# --------------------------------------------------

@app.post(
    "/agent/invoke",
    response_model=AgentResponse
)
def invoke_agent(request: AgentRequest):

    try:

        # Invoke the existing StringAgent
        agent_response = string_agent.invoke(request.prompt)

        return AgentResponse(
            response=str(agent_response)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail="Agent execution failed"
        ) from error
