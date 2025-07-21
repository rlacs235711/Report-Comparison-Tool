from agents import Agent, Runner
from datetime import datetime, timedelta
import json
import ast

def get_availability_parser_agent(prompt: str, use_model: str):
    return Agent(
        name="Availability Parser Agent",
        instructions=prompt,
        model=use_model,
    )

async def extract_OpenAI(prompt: str, text: str, model: str) -> str:
    runner = Runner()
    agent = get_availability_parser_agent(prompt, model)
    result = await runner.run(agent, text)

    return result.final_output