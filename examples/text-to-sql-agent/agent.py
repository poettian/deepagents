import argparse
import os
import sys
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from rich.console import Console
from rich.panel import Panel

from sql_tools import create_sql_tools

# Load environment variables
load_dotenv()

console = Console()


def create_sql_deep_agent():
    """Create and return a text-to-SQL Deep Agent"""

    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Share the OpenRouter model between the agent and SQL query checker.
    model = ChatOpenRouter(
        model=os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash"),
        temperature=0,
        # Let the SDK own and close its clients without attribution headers.
        app_url=None,
        app_title=None,
    )

    sql_tools = create_sql_tools(Path(base_dir) / "chinook.db", model)

    # Create the Deep Agent with all parameters
    agent = create_deep_agent(
        model=model,
        memory=["/AGENTS.md"],  # Agent identity and general instructions
        skills=["/skills/"],  # Query-writing and schema-exploration workflows
        tools=sql_tools,  # SQL database tools
        subagents=[],  # No subagents needed
        backend=FilesystemBackend(root_dir=base_dir, virtual_mode=True),
    )

    return agent


def main():
    """Main entry point for the SQL Deep Agent CLI"""
    parser = argparse.ArgumentParser(
        description="Text-to-SQL Deep Agent powered by LangChain Deep Agents and OpenRouter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py "What are the top 5 best-selling artists?"
  python agent.py "Which employee generated the most revenue by country?"
  python agent.py "How many customers are from Canada?"
        """,
    )
    parser.add_argument(
        "question",
        type=str,
        help="Natural language question to answer using the Chinook database",
    )

    args = parser.parse_args()

    # Display the question
    console.print(
        Panel(f"[bold cyan]Question:[/bold cyan] {args.question}", border_style="cyan")
    )
    console.print()

    # Create the agent
    console.print("[dim]Creating SQL Deep Agent...[/dim]")
    agent = create_sql_deep_agent()

    # Invoke the agent
    console.print("[dim]Processing query...[/dim]\n")

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": args.question}]}
        )

        # Extract and display the final answer
        final_message = result["messages"][-1]
        answer = (
            final_message.content
            if hasattr(final_message, "content")
            else str(final_message)
        )

        console.print(
            Panel(f"[bold green]Answer:[/bold green]\n\n{answer}", border_style="green")
        )

    except Exception as e:
        console.print(
            Panel(f"[bold red]Error:[/bold red]\n\n{str(e)}", border_style="red")
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
