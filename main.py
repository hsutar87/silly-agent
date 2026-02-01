import sys
import asyncio
import os
from pathlib import Path
from colored import fg, attr

from src.agent import SillyAgent
from src.reader import UniversalReader
from src.vector_store import VectorStoreManager


def get_docs_from_data():
    reader = UniversalReader()
    docs = []

    for f in Path("./data").glob("*.*"):
        if f.suffix.lower() in [".pdf", ".docx", ".txt"]:
            doc = reader.read_file(f)
            doc["metadata"]["source"] = f.name
            doc["metadata"]["last_modified"] = int(os.path.getmtime(f))
            docs.append(doc)

    return docs


def run_ingestion(mode="delta"):
    store = VectorStoreManager()
    docs = get_docs_from_data()

    if not docs:
        print("No files found for ingestion in ./data")
        return

    if mode == "rebuild":
        print("🧹 Rebuilding index from scratch...")
        store.rebuild_index(docs)
        print("✅ Rebuild complete.")
    else:
        print("🟢 Running delta update (new/updated files only)...")
        store.update_delta(docs)
        print("✅ Delta update complete.")


async def chat_loop():
    agent = SillyAgent(
        model="deepseek-r1:8b",
        sse_url="http://localhost:8000/sse"
    )
    
    store = VectorStoreManager()

    SYSTEM_PROMPT = """
    You are Silly. A brilliant, sarcastic, and witty librarian.
    - Answer with maximum impact and minimum words (Max 2-3 sentences).
    - You have a memory. If the user refers to previous messages, use your history.
    - You MUST answer ONLY using the CONTEXT provided below.
    - If CONTEXT is empty, say "I have no idea. Ask me something else!".
    - Cite sources as [Source: file.pdf].
    - Be sharp, awesome, and concise.
    """

    print("\n" + "=" * 50)
    print("  SILLY AGENT: WITTY, CONCISE, AND DANGEROUS")
    print("=" * 50)

    while True:
        try:
            query = input("\n💬 You: ").strip()
            if not query:
                continue
            if query.lower() in ["quit", "exit"]:
                break
            if query.lower() == "clear":
                agent.memory.clear()
                print("🧹 *Memory wiped.*")
                continue

            results = store.search(query, limit=3)

            if not results:
                print(f"{fg('red')}No relevant documents found. Try another question.{attr('reset')}")
                continue

            context_text = ""
            sources = []
            for r in results:
                context_text += r["content"] + "\n"
                sources.append(r.get("source", "Unknown"))

            sources = sorted(set(sources))
            sources_str = ", ".join(sources)

            context_prompt = (
                f"CONTEXT:\n{context_text}\n\n"
                f"Sources: {sources_str}\n"
            )

            print("\n💃 Silly: ", end="", flush=True)

            async for chunk in agent.run(query, SYSTEM_PROMPT + "\n" + context_prompt):
                print(chunk, end="", flush=True)

            print("\n" + "─" * 30)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    if "--ingest" in sys.argv:
        run_ingestion(mode="rebuild")
    else:
        asyncio.run(chat_loop())
