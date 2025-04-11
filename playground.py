from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.playground import Playground, serve_playground_app
from agno.storage.sqlite import SqliteStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.mcp import MCPTools
from mcp import StdioServerParameters   
agent_storage: str = "tmp/agents.db"

web_agent = Agent(
    name="Web Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions=["Always include sources"],
    # Store the agent sessions in a sqlite database
    storage=SqliteStorage(table_name="web_agent", db_file=agent_storage),
    # Adds the current date and time to the instructions
    add_datetime_to_instructions=True,
    # Adds the history of the conversation to the messages
    add_history_to_messages=True,
    # Number of history responses to add to the messages
    num_history_responses=5,
    # Adds markdown formatting to the messages
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Always use tables to display data"],
    storage=SqliteStorage(table_name="finance_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

browser_agent = Agent(
    name="Browser Agent",
    model=OpenAIChat(id="gpt-4o"),
    description="An agent that MUST use browser tools to interact with the web. It cannot refuse browsing tasks.",
    tools=[
        MCPTools(
            server_params=StdioServerParameters(
                name="playwright",
                command="npx",
                args=["@playwright/mcp@latest"],
            )
        )
    ],
    instructions=[
        "You are a web browsing assistant.",
        "You MUST use the available browser tools (like navigate, click, type, snapshot) to interact with web pages as requested by the user.",
        "DO NOT refuse requests to browse or interact with web pages. Use the tools provided.",
        "When asked to open a specific URL, you MUST use the 'navigate' tool with the provided URL."
    ],
    storage=SqliteStorage(table_name="browser_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
    debug_mode=True,
    show_tool_calls=True,
)

app = Playground(agents=[web_agent, finance_agent, browser_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)