import sys
import asyncio
from pathlib import Path
from src.agent import SillyAgent
from src.reader import UniversalReader
from src.vector_store import VectorStoreManager

def run_ingestion():
    reader = UniversalReader()
    store = VectorStoreManager()
    for f in Path("./data").glob("*.*"):
        if f.suffix.lower() in ['.pdf', '.docx', '.txt']:
            print(f"✨ Ingesting: {f.name}")
            doc = reader.read_file(f)
            store.ingest(doc["content"], doc["metadata"])

async def chat_loop():
    agent = SillyAgent(model="deepseek-r1:8b")
    
    SYSTEM_PROMPT = """
    You are Silly. A brilliant, sarcastic, and witty librarian.
    - Answer with maximum impact and minimum words (Max 2-3 sentences).
    - You have a memory. If the user refers to previous messages, use your history.
    - Cite sources as [Source: file.pdf].
    - Be sharp, awesome, and concise.
    """

    print("\n" + "="*50)
    print("  SILLY AGENT: WITTY, CONCISE, AND DANGEROUS")
    print("="*50)
    
    while True:
        try:
            query = input("\n💬 You: ").strip()
            if not query: continue
            if query.lower() in ['quit', 'exit']: break
            if query.lower() == 'clear': 
                agent.memory.clear()
                print("🧹 *Memory wiped.*")
                continue

            print("\n💃 Silly: ", end="", flush=True)
            
            async for chunk in agent.run(query, SYSTEM_PROMPT):
                if any(emoji in chunk for emoji in ["🔍", "🛠️", "🧠"]):
                    print(f"\n{chunk}", end="", flush=True)
                else:
                    print(chunk, end="", flush=True)
            
            print("\n" + "─"*30)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    if "--ingest" in sys.argv:
        run_ingestion()
    else:
        asyncio.run(chat_loop())