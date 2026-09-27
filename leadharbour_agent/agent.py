"""An optional ADK agent that scores and asks for review, never sends."""

import os

from google.adk.agents import Agent

from leadharbour.agent_tools import request_human_review, score_lead

model = os.getenv("LEADHARBOUR_GEMINI_MODEL", "gemini-2.5-flash")
if os.getenv("LEADHARBOUR_OLLAMA_MODEL"):
    from google.adk.models.lite_llm import LiteLlm
    model = LiteLlm(model=f"ollama_chat/{os.environ['LEADHARBOUR_OLLAMA_MODEL']}")

root_agent = Agent(
    name="leadharbour_review_agent",
    model=model,
    instruction=(
        "You explain a historical propensity score. Use score_lead only when all five "
        "pre-contact fields are provided. Never claim causal uplift or decide that a "
        "person should be contacted. Do not send email, contact a CRM, or invent lead "
        "details. If asked to prepare outreach, call request_human_review and explain "
        "that a person must check consent, evidence and wording first."
    ),
    tools=[score_lead, request_human_review],
)
