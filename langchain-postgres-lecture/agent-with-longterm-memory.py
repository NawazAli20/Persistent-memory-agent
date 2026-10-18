"""Persist a conversation in PostgreSQL; optionally display its saved history."""
import argparse
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-id", default="1")
    parser.add_argument("--history", action="store_true",
                        help="Display saved messages without invoking the model")
    args = parser.parse_args()

    db_uri = os.getenv("DATABASE_URL")
    if not db_uri:
        raise SystemExit("Set DATABASE_URL in .env.")
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("Set GROQ_API_KEY in .env.")

    model = init_chat_model(
        model="openai/gpt-oss-120b",
        model_provider="groq",
    )
    thread_config = {"configurable": {"thread_id": args.thread_id}}

    with PostgresSaver.from_conn_string(db_uri) as checkpointer:
        checkpointer.setup()
        agent = create_agent(
            model=model,
            tools=[],
            checkpointer=checkpointer,
            system_prompt="You are a helpful assistant.",
        )

        print(f"Thread: {args.thread_id}")
        if not args.history:
            user_query = input("You: ").strip()
            if not user_query:
                raise SystemExit("Enter a nonempty message.")
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_query)]},
                config=thread_config,
            )
            print("AI:", result["messages"][-1].content)

        # Read the latest state while the database connection is still open.
        snapshot = agent.get_state(thread_config)
        messages = snapshot.values.get("messages", [])
        print("\nChat history:")
        if not messages:
            print("(No saved messages for this thread.)")
        for message in messages:
            print(f"{message.type}: {message.content}")


if __name__ == "__main__":
    main()
