![AI Agents That Remember — LangChain and PostgreSQL](assets/youtube-banner.png)

# Persistent Agent Memory with LangChain and PostgreSQL

Build a Python agent that remembers a conversation after the program exits. In this lecture, you will install PostgreSQL, manage Python dependencies with **uv**, connect a Groq model, and retrieve saved chat messages.

**Lecture focus: `PostgresSaver`.** We use the phrase *persistent conversation memory* because the database preserves a thread across restarts.

## Saver vs. Store

| | `PostgresSaver` | `PostgresStore` |
|---|---|---|
| Agent argument | `checkpointer=...` | `store=...` |
| Purpose | Save and restore conversation state | Store selected facts or preferences |
| Scope | One conversation identified by `thread_id` | Namespaces and keys, often grouped by user |
| Memory behavior | Checkpoints are written during agent execution | Your tools or application explicitly write and retrieve memories |
| Example | Resume yesterday's conversation | Reuse a user's preferences in a new conversation |

LangChain calls thread-scoped history **short-term memory**, even when it survives restarts in PostgreSQL. Cross-thread facts are **long-term memory**. Passing a store alone does not make the agent remember every message. Both components can be used together. See [short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory) and [long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory).

## What you need

- macOS or Windows, with terminal access.
- PostgreSQL 17 server running locally.
- uv and Python 3.12 or newer; these examples use Python 3.13.
- A [Groq API key](https://console.groq.com/keys) and access to `openai/gpt-oss-120b`. Check [Groq's model list](https://console.groq.com/docs/models) if availability changes.

This example uses `tools=[]`. Weather, Tavily, Ollama, and fallback models are omitted because they are not used in this lesson.

> Installing `langgraph-checkpoint-postgres` and `psycopg[binary]` installs Python integration packages, **not the PostgreSQL server**. Install the server separately below.

## 1. Install PostgreSQL

### macOS: Homebrew

Install [Homebrew](https://brew.sh/) first if needed. In Terminal:

```bash
brew install postgresql@17
brew services start postgresql@17
export PATH="$(brew --prefix postgresql@17)/bin:$PATH"
pg_isready -h localhost -p 5432
```

Look for `accepting connections`. Homebrew initializes a database cluster and the service starts now and at login. To make the command path available in future zsh terminals, run once:

```bash
echo 'export PATH="$(brew --prefix postgresql@17)/bin:$PATH"' >> ~/.zshrc
```

Create the lecture database:

```bash
createdb -h localhost -U "$(whoami)" agent_memory
psql -h localhost -U "$(whoami)" -d agent_memory -c "SELECT current_database(), current_user;"
```

A fresh default Homebrew installation normally uses your macOS username as the initial database administrator. It does not automatically provide a `postgres/postgres` login. The local passwordless connection below assumes the default local authentication settings.

If Homebrew reports **no bottle available**, inspect `brew config` and `brew doctor`. A bottle is a prebuilt package; availability depends on your system. If you choose to build from source and have Xcode Command Line Tools installed:

```bash
brew install --build-from-source postgresql@17
```

A source build can take longer and may fail on unsupported systems. Start the service only after installation succeeds. See the [Homebrew PostgreSQL 17 formula](https://formulae.brew.sh/formula/postgresql@17).

### Windows: PostgreSQL installer

1. Download the PostgreSQL 17 Windows installer through the [official PostgreSQL download page](https://www.postgresql.org/download/windows/).
2. Install **PostgreSQL Server**, **Command Line Tools**, and optionally **pgAdmin**.
3. Record the password you choose for the `postgres` database user.
4. Keep port `5432` unless another application already uses it. Stack Builder add-ons are not needed for this lecture.
5. Finish installation. Open Windows **Services** and confirm the PostgreSQL service is running; start it if necessary.

In **PowerShell**, use the binaries directly. Adjust the folder if you selected another installation directory or version:

```powershell
& "C:\Program Files\PostgreSQL\17\bin\pg_isready.exe" -h localhost -p 5432
& "C:\Program Files\PostgreSQL\17\bin\createdb.exe" -h localhost -U postgres -W agent_memory
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -h localhost -U postgres -d agent_memory -W -c "SELECT current_database(), current_user;"
```

Enter the installer password when prompted. Expected database: `agent_memory`; expected user: `postgres`.

## 2. Install uv and Python dependencies

On macOS:

```bash
brew install uv
```

On Windows, in PowerShell:

```powershell
winget install --id=astral-sh.uv -e
```

Open a new terminal if `uv` is not found. These installation methods are documented in the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

Open the repository folder, then run on either platform:

```bash
uv python install 3.13
uv sync
```

The included `pyproject.toml` declares the dependencies. `uv sync` creates the environment and a lockfile. Commit `uv.lock` after resolving dependencies so students can share the same versions.

If adapting this lesson in your own project, run `uv init` only if it has no `pyproject.toml`, then add:

```bash
uv add langchain langchain-groq python-dotenv langgraph-checkpoint-postgres "psycopg[binary]"
```

The quoted `psycopg[binary]` extra installs the binary distribution of the PostgreSQL Python driver. You do not need to activate the virtual environment when using `uv run`. See [uv projects](https://docs.astral.sh/uv/guides/projects/).

## 3. Configure the environment

Copy `.env.example` to `.env` in the repository root.

macOS:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env` with your actual values.

**macOS example** — replace `YOUR_MAC_USERNAME` with the output of `whoami`:

```dotenv
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://YOUR_MAC_USERNAME@localhost:5432/agent_memory
```

For the instructor's local account, the URL is `postgresql://nali@localhost:5432/agent_memory`.

**Windows example** — use the password chosen during installation:

```dotenv
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/agent_memory
```

Percent-encode reserved characters in the password portion of a URL (for example, `@` becomes `%40`). Do not commit `.env`; the included `.gitignore` excludes it.

`load_dotenv()` loads these values. You do not need to reassign them with `os.environ["KEY"] = os.getenv("KEY")`, which fails when the value is missing.

## 4. The Saver example

The included [agent-with-longterm-memory.py](agent-with-longterm-memory.py) preserves the original lecture filename, but demonstrates **persistent thread-scoped memory**. It adds optional `--thread-id` and `--history` arguments to the starter code:

```python
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
```

The key operations are:

1. `PostgresSaver.from_conn_string(...)` opens the database connection.
2. `setup()` creates the required tables and applies migrations. This lecture calls it at startup; production deployments can run it during provisioning.
3. `checkpointer=checkpointer` attaches persistence to the agent.
4. The same `thread_id` resumes the same conversation.
5. `get_state()` reconstructs the current saved messages.

Keep database-backed operations inside the `with` block. Send only the new message to `invoke()`; the checkpointer restores previous state. See [LangChain's PostgreSQL example](https://docs.langchain.com/oss/python/langchain/short-term-memory).

## 5. Run the agent and test persistence

First run:

```bash
uv run agent-with-longterm-memory.py --thread-id 1
```

Enter:

```text
You: My name is Nawaz.
```

The program replies, prints the saved history, and exits. Run the same command again:

```bash
uv run agent-with-longterm-memory.py --thread-id 1
```

Enter:

```text
You: What is my name?
```

An expected answer is “Your name is Nawaz.” Exact model wording varies. The printed history should include the earlier introduction.

Start a separate conversation:

```bash
uv run agent-with-longterm-memory.py --thread-id 2
```

Thread `2` does not inherit thread `1`'s messages. Return to `--thread-id 1` to resume the original conversation. Old conversations from `InMemorySaver`, or from a Store with no writes, are not automatically migrated.

## 6. Retrieve the actual saved chat

To read the stored transcript without sending another prompt:

```bash
uv run agent-with-longterm-memory.py --thread-id 1 --history
```

This initializes the agent locally and reads its saved state; it does **not** invoke the model or make a generation request. The script still expects the configured environment variables. For a thread without messages, it prints a clear empty-history message.

The essential retrieval code is:

```python
snapshot = agent.get_state(thread_config)

for message in snapshot.values.get("messages", []):
    print(f"{message.type}: {message.content}")
```

Typical output:

```text
Chat history:
human: My name is Nawaz.
ai: Nice to meet you, Nawaz!
human: What is my name?
ai: Your name is Nawaz.
```

These are the saved message objects, not a model-generated summary. If tools are added later, the state can also contain tool messages. If you later trim or summarize state, the latest snapshot reflects those changes rather than an immutable full transcript.

## 7. Inspect PostgreSQL directly

Connect on macOS:

```bash
psql -h localhost -U "$(whoami)" -d agent_memory
```

Connect on Windows PowerShell:

```powershell
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -h localhost -U postgres -d agent_memory -W
```

Then run these commands at the PostgreSQL prompt:

```sql
-- List tables.
\dt

-- List conversations with saved checkpoints.
SELECT DISTINCT thread_id
FROM checkpoints
ORDER BY thread_id;

-- Inspect recent checkpoint records for thread 1.
SELECT thread_id, checkpoint_ns, checkpoint_id
FROM checkpoints
WHERE thread_id = '1'
ORDER BY checkpoint_id DESC
LIMIT 5;
```

Typical tables include `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, and `checkpoint_migrations`; exact schema details depend on the package version. Message payloads can be serialized across tables. Prefer `agent.get_state()` for readable chat history rather than decoding database internals.

Exit with `\q`. Windows users can also open **pgAdmin**, connect to their local server, select `agent_memory`, and run the SELECT statements in Query Tool. The `\dt` and `\q` commands are specific to psql.

## Troubleshooting

| Symptom | What to check |
|---|---|
| Connection refused | PostgreSQL must be installed and running on the host/port in `DATABASE_URL`. |
| Database does not exist | Create `agent_memory` using `createdb`. |
| Role does not exist | Homebrew commonly uses your macOS username; the Windows installer uses `postgres`. |
| Password authentication failed | Check the password and URL encoding. |
| Agent cannot recall your name | Use `checkpointer=`, reuse the same thread ID and database, and introduce yourself after enabling Saver. |
| SystemMessage validation error | Use a plain string for `system_prompt`. |
| No user message reaches the model | The input key is `messages`, not `messgaes`. |
| Groq authentication, model, or rate-limit error | Check the API key, model availability, and account limits. |
| No bottle available | Inspect `brew config` and `brew doctor`; a source build is an optional fallback. |

This is a local teaching example. Thread IDs separate state; they are not authorization controls. Chat history is stored locally in PostgreSQL, and conversation context is sent to Groq when you invoke the model. Large conversations eventually need a context-management strategy.

## Repository contents

- `README.md`: lecture guide.
- `agent-with-longterm-memory.py`: one-turn chat plus history retrieval.
- `pyproject.toml`: dependencies managed by uv.
- `.env.example`: environment template.
- `.gitignore`: excludes credentials and local environments.
- `assets/youtube-banner.png`: YouTube-style lecture banner, also shown above.
- `assets/banner-prompt.txt`: image-generation prompt and tool provenance.

Upload the README and the `assets` folder together so the relative image link renders on GitHub.
