import json
import requests
from typing import Dict, Any, AsyncGenerator
from mcp import ClientSession
from mcp.client.sse import sse_client
from .memory import ConversationManager

class SillyAgent:
    def __init__(self, model: str, sse_url: str = "http://localhost:8000/sse"):
        self.model = model
        self.ollama_url = "http://localhost:11434/api/chat"
        self.sse_url = sse_url
        self.tools = []
        self.memory = ConversationManager()

    async def _get_tools(self):
        async with sse_client(self.sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_res = await session.list_tools()
                return [{"name": t.name, "description": t.description, "parameters": t.inputSchema} for t in tools_res.tools]

    async def _call_tool(self, name: str, args: Dict):
        async with sse_client(self.sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.call_tool(name, args)
                return res.content[0].text

    async def run(self, user_input: str, system_prompt: str) -> AsyncGenerator[str, None]:
        if not self.tools:
            yield "🔍 *Checking the toolbelt...*"
            self.tools = await self._get_tools()

        # Build messages with history
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.memory.get_history())
        messages.append({"role": "user", "content": user_input})

        # 1. Decision Phase
        eval_payload = {
            "model": self.model,
            "messages": messages + [{"role": "system", "content": f"Tools: {json.dumps(self.tools)}. Return JSON: {{\"tool\": \"name\", \"args\": {{...}}}} or {{\"tool\": \"none\"}}"}],
            "stream": False,
            "format": "json"
        }
        
        res = requests.post(self.ollama_url, json=eval_payload).json()
        decision = json.loads(res['message']['content'])

        context = ""
        if decision.get("tool") and decision["tool"] != "none":
            yield f"🛠️ *Digging through: {decision['tool']}...*"
            context = await self._call_tool(decision["tool"], decision["args"])

        # 2. Final Answer Phase
        final_messages = messages + [{"role": "system", "content": f"CONTEXT: {context}"}]
        yield "🧠 *Thinking...*\n"
        
        response = requests.post(self.ollama_url, json={"model": self.model, "messages": final_messages, "stream": True}, stream=True)
        
        full_answer = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode("utf-8"))
                if not chunk.get("done"):
                    content = chunk["message"]["content"]
                    full_answer += content
                    yield content

        # Save to memory
        self.memory.add_message("user", user_input)
        self.memory.add_message("assistant", full_answer)