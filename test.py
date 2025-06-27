from agno.agent import Agent
from agno.tools.exa import ExaTools
from agno.models.google import Gemini
from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    model=Gemini(id="gemini-2.0-flash-lite"),  # This model supports multimodal
    tools=[ExaTools(
        include_domains=["cnbc.com", "reuters.com", "bloomberg.com"],
        category="news",
        text_length_limit=1000,
    )],
    show_tool_calls=True,
)
agent.print_response("Search for AAPL news", markdown=True)