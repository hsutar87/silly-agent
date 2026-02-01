import json
import requests
from typing import Dict, Any, AsyncGenerator, List
from colored import fg, attr
from mcp import ClientSession
from mcp.client.sse import sse_client
from utils.conversation import ConversationManager
from pprint import pformat


class SillyAgent:
    def __init__(self, model: str, sse_url: str = "http://localhost:8000/sse"):
        """
        Initialize the SillyAgent with MCP connection details.

        :param model: The model to use for communication
        :param sse_url: The URL for MCP SSE server (default: "http://localhost:8000/sse")
        """
        self.model = model
        self.ollama_url = "http://localhost:11434/api/chat"
        self.sse_url = sse_url
        self.tools: List[Dict[str, Any]] = []  # List of available tools
        self.memory = ConversationManager()

    async def _get_tools(self) -> AsyncGenerator[str, None]:
        """
        Fetch available tools from the MCP server.

        :return: List of tools
        """
        async with sse_client(self.sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_res = await session.list_tools()
                # Yield tools fetched for debugging purposes
                yield f"{fg('cyan')}🔧 Tools fetched from MCP server:{attr('reset')}\n"
                yield f"{fg('cyan')}{pformat(tools_res.tools)}{attr('reset')}\n"
                for tool in tools_res.tools:
                    yield {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }

    async def _call_tool(self, name: str, args: Dict = None) -> AsyncGenerator[str, None]:
        """
        Calls a specific tool on the MCP server and retrieves the response.

        :param name: Name of the tool to call
        :param args: Arguments for the tool
        :return: Tool response
        """
        yield f"{fg('yellow')}🛠️ Calling tool: {name}{attr('reset')}\n"
        yield f"{fg('yellow')}Arguments: {pformat(args)}{attr('reset')}\n"

        async with sse_client(self.sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.call_tool(name, args)
                # Yield tool response for debugging
                # yield f"{fg('magenta')}📦 Tool response for {name}: {attr('reset')}\n"
                # yield f"{fg('magenta')}{pformat(res.content)}{attr('reset')}\n"
                # Yielding the tool result text
                yield res.content[0].text

    async def run(self, user_input: str, system_prompt: str) -> AsyncGenerator[str, None]:
        """
        Main loop for interacting with the agent and processing the user input.

        :param user_input: The input message from the user
        :param system_prompt: System level context to guide the agent
        :yield: The response from the assistant
        """
        # Fetch tools if not already fetched
        if not self.tools:
            yield f"{fg('cyan')}🔍 Checking the toolbelt...{attr('reset')}\n"
            async for tool in self._get_tools():
                if isinstance(tool, str):  # It's a debug string, so print it
                    yield tool
                else:
                    self.tools.append(tool)

        # Prepare the messages for the Ollama model
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.memory.get_history())
        messages.append({"role": "user", "content": user_input})

        # Create the payload for evaluating the request
        eval_payload = {
            "model": self.model,
            "messages": messages + [
                {
                    "role": "system",
                    "content": (
                        f"Tools: {json.dumps(self.tools)}. "
                        "Return JSON: "
                        "{\"tool\": \"name\", \"args\": {...}} "
                        "or {\"tool\": \"none\"}"
                    ),
                }
            ],
            "stream": False,
            "format": "json",
        }

        # Get the response from Ollama model
        res = requests.post(self.ollama_url, json=eval_payload).json()
        reasoning = res.get("message", {}).get("reasoning")

        # If reasoning is available, yield it
        if reasoning:
            yield f"{fg('cyan')}{attr('bold')}🧠 Reasoning:{attr('reset')}\n"
            yield f"{fg('cyan')}{reasoning}{attr('reset')}\n\n"

        # Parse the tool decision and arguments from the response
        decision = json.loads(res["message"]["content"])
        tool_name = decision.get("tool")
        tool_args = decision.get("args", {})

        yield f"{fg('yellow')}💡 Decision: Tool selected: {tool_name}{attr('reset')}\n"
        yield f"{fg('yellow')}Arguments for tool: {pformat(tool_args)}{attr('reset')}\n"

        tool_output = None

        # If tool is to be used, run it and fetch the output
        if tool_name and tool_name != "none":
            yield f"{fg('yellow')}🛠️ Running tool: {tool_name}{attr('reset')}\n"
            try:
                async for tool_output_chunk in self._call_tool(tool_name, tool_args or None):
                    yield tool_output_chunk  # Yielding each chunk of the tool response

                # Now, ensure tool_output is properly accumulated
                tool_output = tool_output_chunk

                # Try parsing the tool output as JSON
                try:
                    tool_output = json.loads(tool_output)
                except Exception:
                    pass

                yield f"{fg('magenta')}{attr('bold')}\n📦 Tool Output:{attr('reset')}\n"
                yield f"{fg('magenta')}{pformat(tool_output)}{attr('reset')}\n\n"
            except Exception as e:
                yield f"{fg('red')}❌ Tool error: {e}{attr('reset')}\n"

        # Finalize the messages with tool output and prepare for final response
        final_messages = messages + [
            {"role": "system", "content": f"TOOL_OUTPUT: {json.dumps(tool_output)}"}
        ]

        yield f"{fg('blue')}{attr('bold')}🤖 Assistant:{attr('reset')}\n"

        # Get the final response from Ollama model, streaming if necessary
        response = requests.post(
            self.ollama_url,
            json={"model": self.model, "messages": final_messages, "stream": True},
            stream=True,
        )

        full_answer = ""
        for line in response.iter_lines():
            if not line:
                continue

            chunk = json.loads(line.decode("utf-8"))
            if chunk.get("done"):
                continue

            message = chunk.get("message", {})

            # If reasoning is available in stream, yield it
            if "reasoning" in message:
                yield f"{fg('cyan')}{attr('bold')}🧠 Reasoning (stream):{attr('reset')}\n"
                yield f"{fg('cyan')}{message['reasoning']}{attr('reset')}"
                continue

            # Process the content and yield the answer
            content = message.get("content", "")
            if content:
                full_answer += content
                yield f"{fg('green')}{content}{attr('reset')}"

        # Store the user and assistant messages in memory
        self.memory.add_message("user", user_input)
        self.memory.add_message("assistant", full_answer)
