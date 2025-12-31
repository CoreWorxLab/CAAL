"""Simplified Ollama LLM Node with think parameter support.

This module provides a custom llm_node implementation that bypasses LiveKit's
LLM wrapper to enable direct Ollama calls with the `think` parameter.

Key Features:
- Direct Ollama API calls with think=False for lower latency (Qwen3)
- Tool discovery from @function_tool methods and MCP servers
- Tool execution routing (agent methods, n8n workflows, MCP tools)
- Streaming responses for best UX

Usage:
    class MyAgent(Agent):
        async def llm_node(self, chat_ctx, tools, model_settings):
            async for chunk in ollama_llm_node(
                self, chat_ctx, model="qwen3:8b", think=False
            ):
                yield chunk
"""

import inspect
import json
import logging
import time
from collections.abc import AsyncIterable
from typing import Any

import ollama

from ..utils.formatting import strip_markdown_for_tts
from ..integrations.n8n import execute_n8n_workflow

logger = logging.getLogger(__name__)


class ToolDataCache:
    """Caches recent tool response data for context injection.

    Tool responses often contain structured data (IDs, arrays) that the LLM
    needs for follow-up calls. This cache preserves that data separately
    from chat history and injects it into context on each LLM call.
    """

    def __init__(self, max_entries: int = 3):
        self.max_entries = max_entries
        self._cache: list[dict] = []

    def add(self, tool_name: str, data: Any) -> None:
        """Add tool response data to cache."""
        entry = {"tool": tool_name, "data": data, "timestamp": time.time()}
        self._cache.append(entry)
        if len(self._cache) > self.max_entries:
            self._cache.pop(0)  # Remove oldest

    def get_context_message(self) -> str | None:
        """Format cached data as context string for LLM injection."""
        if not self._cache:
            return None
        parts = ["Recent tool response data for reference:"]
        for entry in self._cache:
            parts.append(f"\n{entry['tool']}: {json.dumps(entry['data'])}")
        return "\n".join(parts)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


class OllamaLLMNode:
    """Encapsulates Ollama LLM configuration and tool handling."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        think: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
    ):
        self.model = model
        self.think = think
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self._tools_cache: list[dict] | None = None

    def _get_ollama_options(self) -> dict:
        """Get Ollama options dict."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }


async def ollama_llm_node(
    agent,
    chat_ctx,
    model: str = "qwen3:8b",
    think: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    num_ctx: int = 8192,
    tool_data_cache: ToolDataCache | None = None,
    max_turns: int = 20,
) -> AsyncIterable[str]:
    """Custom LLM node using Ollama directly with think parameter support.

    This function should be called from an Agent's llm_node method override.

    Args:
        agent: The Agent instance (self)
        chat_ctx: Chat context from LiveKit
        model: Ollama model name
        think: Enable thinking mode (False for lower latency with Qwen3)
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling limit
        num_ctx: Context window size
        tool_data_cache: Cache for structured tool response data
        max_turns: Max conversation turns to keep in sliding window

    Yields:
        String chunks for TTS output

    Example:
        class MyAgent(Agent):
            async def llm_node(self, chat_ctx, tools, model_settings):
                async for chunk in ollama_llm_node(self, chat_ctx, think=False):
                    yield chunk
    """
    options = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "num_ctx": num_ctx,
    }

    try:
        # Build messages from chat context with sliding window
        messages = _build_messages_from_context(
            chat_ctx,
            tool_data_cache=tool_data_cache,
            max_turns=max_turns,
        )

        # Discover tools from agent and MCP servers
        ollama_tools = await _discover_tools(agent)

        # If tools available, check for tool calls first (non-streaming)
        if ollama_tools:
            response = ollama.chat(
                model=model,
                messages=messages,
                tools=ollama_tools,
                think=think,
                stream=False,
                options=options,
            )

            if hasattr(response, "message"):
                # Check for tool calls
                if hasattr(response.message, "tool_calls") and response.message.tool_calls:
                    tool_calls = response.message.tool_calls
                    logger.info(f"Ollama returned {len(tool_calls)} tool call(s)")

                    # Track tool usage for frontend indicator
                    tool_names = [tc.function.name for tc in tool_calls]
                    tool_params = [tc.function.arguments for tc in tool_calls]

                    # Publish tool status immediately (bullshit detector!)
                    if hasattr(agent, "_on_tool_status") and agent._on_tool_status:
                        import asyncio
                        asyncio.create_task(agent._on_tool_status(True, tool_names, tool_params))

                    # Execute tools and get results (cache structured data)
                    messages = await _execute_tool_calls(
                        agent, messages, tool_calls, response.message,
                        tool_data_cache=tool_data_cache,
                    )

                    # Stream follow-up response with tool results
                    followup = ollama.chat(
                        model=model,
                        messages=messages,
                        think=think,
                        stream=True,
                        options=options,
                    )

                    for chunk in followup:
                        if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                            if chunk.message.content:
                                yield strip_markdown_for_tts(chunk.message.content)
                    return

                # No tool calls - return content directly
                elif hasattr(response.message, "content") and response.message.content:
                    # Publish no-tool status immediately
                    if hasattr(agent, "_on_tool_status") and agent._on_tool_status:
                        import asyncio
                        asyncio.create_task(agent._on_tool_status(False, [], []))
                    yield strip_markdown_for_tts(response.message.content)
                    return

        # No tools or no tool calls - stream directly
        # Publish no-tool status immediately
        if hasattr(agent, "_on_tool_status") and agent._on_tool_status:
            import asyncio
            asyncio.create_task(agent._on_tool_status(False, [], []))

        response_stream = ollama.chat(
            model=model,
            messages=messages,
            tools=None,
            think=think,
            stream=True,
            options=options,
        )

        for chunk in response_stream:
            if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                if chunk.message.content:
                    yield strip_markdown_for_tts(chunk.message.content)

    except Exception as e:
        logger.error(f"Error in ollama_llm_node: {e}", exc_info=True)
        yield f"I encountered an error: {e}"


def _build_messages_from_context(
    chat_ctx,
    tool_data_cache: ToolDataCache | None = None,
    max_turns: int = 20,
) -> list[dict]:
    """Build Ollama messages with sliding window and tool data context.

    Message order:
    1. System prompt (always first, never trimmed)
    2. Tool data context (injected from cache)
    3. Chat history (sliding window applied)

    Args:
        chat_ctx: LiveKit chat context
        tool_data_cache: Cache of recent tool response data
        max_turns: Max conversation turns to keep (1 turn = user + assistant)
    """
    system_prompt = None
    chat_messages = []

    for item in chat_ctx.items:
        item_type = type(item).__name__

        if item_type == "ChatMessage":
            msg = {"role": item.role, "content": item.text_content}
            if item.role == "system":
                system_prompt = msg
            else:
                chat_messages.append(msg)
        elif item_type == "FunctionCall":
            try:
                chat_messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": item.id,
                        "function": {
                            "name": item.name,
                            "arguments": getattr(item, "arguments", {}) or {},
                        },
                    }],
                })
            except AttributeError:
                pass
        elif item_type == "FunctionCallOutput":
            try:
                chat_messages.append({
                    "role": "tool",
                    "content": str(item.content),
                    "tool_call_id": item.tool_call_id,
                })
            except AttributeError:
                pass

    # Build final message list
    messages = []

    # 1. System prompt always first
    if system_prompt:
        messages.append(system_prompt)

    # 2. Inject tool data context
    if tool_data_cache:
        context = tool_data_cache.get_context_message()
        if context:
            messages.append({"role": "system", "content": context})

    # 3. Apply sliding window to chat history
    # max_turns * 2 accounts for user + assistant pairs
    max_messages = max_turns * 2
    if len(chat_messages) > max_messages:
        trimmed = len(chat_messages) - max_messages
        chat_messages = chat_messages[-max_messages:]
        logger.debug(f"Sliding window: trimmed {trimmed} old messages")

    messages.extend(chat_messages)
    return messages


async def _discover_tools(agent) -> list[dict] | None:
    """Discover tools from agent methods and MCP servers.

    Tools are cached on the agent instance after first discovery to avoid
    redundant MCP API calls on every user utterance.
    """
    # Return cached tools if available
    if hasattr(agent, "_ollama_tools_cache") and agent._ollama_tools_cache is not None:
        return agent._ollama_tools_cache

    ollama_tools = []

    # Get @function_tool decorated methods from agent
    if hasattr(agent, "_tools") and agent._tools:
        for tool in agent._tools:
            if hasattr(tool, "__func__"):
                func = tool.__func__
                name = func.__name__
                description = func.__doc__ or ""
                sig = inspect.signature(func)
                properties = {}
                required = []

                for param_name, param in sig.parameters.items():
                    if param_name == "self":
                        continue
                    param_type = "string"
                    if param.annotation != inspect.Parameter.empty:
                        if param.annotation == str:
                            param_type = "string"
                        elif param.annotation == int:
                            param_type = "integer"
                        elif param.annotation == float:
                            param_type = "number"
                        elif param.annotation == bool:
                            param_type = "boolean"
                    properties[param_name] = {"type": param_type}
                    if param.default == inspect.Parameter.empty and param_name != "self":
                        required.append(param_name)

                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                })

    # Get MCP tools from all configured servers (except n8n)
    # n8n uses webhook-based workflow discovery, not direct MCP tools
    if hasattr(agent, "_caal_mcp_servers") and agent._caal_mcp_servers:
        for server_name, server in agent._caal_mcp_servers.items():
            # Skip n8n - it uses workflow discovery via _n8n_workflow_tools
            if server_name == "n8n":
                continue

            mcp_tools = await _get_mcp_tools(server)
            # Prefix tools with server name to avoid collisions
            for tool in mcp_tools:
                original_name = tool["function"]["name"]
                tool["function"]["name"] = f"{server_name}__{original_name}"
            ollama_tools.extend(mcp_tools)
            if mcp_tools:
                logger.info(f"Added {len(mcp_tools)} tools from MCP server: {server_name}")

    # Add n8n workflow tools (webhook-based execution, separate from MCP)
    if hasattr(agent, "_n8n_workflow_tools") and agent._n8n_workflow_tools:
        ollama_tools.extend(agent._n8n_workflow_tools)

    # Cache tools on agent and return
    result = ollama_tools if ollama_tools else None
    agent._ollama_tools_cache = result

    return result


async def _get_mcp_tools(mcp_server) -> list[dict]:
    """Get tools from an MCP server in Ollama format."""
    tools = []

    if not mcp_server or not hasattr(mcp_server, "_client") or not mcp_server._client:
        return tools

    try:
        tools_result = await mcp_server._client.list_tools()
        if hasattr(tools_result, "tools"):
            for mcp_tool in tools_result.tools:
                # Convert MCP schema to Ollama format
                parameters = {"type": "object", "properties": {}, "required": []}
                if hasattr(mcp_tool, "inputSchema") and mcp_tool.inputSchema:
                    schema = mcp_tool.inputSchema
                    if isinstance(schema, dict):
                        parameters = schema
                    elif hasattr(schema, "properties"):
                        parameters["properties"] = schema.properties or {}
                        parameters["required"] = getattr(schema, "required", []) or []

                tools.append({
                    "type": "function",
                    "function": {
                        "name": mcp_tool.name,
                        "description": getattr(mcp_tool, "description", "") or "",
                        "parameters": parameters,
                    },
                })

        if tools:
            tool_names = [t["function"]["name"] for t in tools]
            logger.info(f"Loaded {len(tools)} MCP tools: {', '.join(tool_names)}")

    except Exception as e:
        logger.warning(f"Error getting MCP tools: {e}")

    return tools


async def _execute_tool_calls(
    agent,
    messages: list[dict],
    tool_calls: list,
    response_message: Any,
    tool_data_cache: ToolDataCache | None = None,
) -> list[dict]:
    """Execute tool calls and append results to messages.

    Args:
        agent: The agent instance
        messages: Current message list to append to
        tool_calls: List of tool calls from LLM response
        response_message: The original LLM response message
        tool_data_cache: Optional cache to store structured tool response data
    """

    # Add assistant message with tool calls
    tool_call_message = {
        "role": "assistant",
        "content": getattr(response_message, "content", "") or "",
        "tool_calls": [],
    }

    for tool_call in tool_calls:
        tool_call_message["tool_calls"].append({
            "id": getattr(tool_call, "id", ""),
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments or {},
            },
        })

    messages.append(tool_call_message)

    # Execute each tool
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = tool_call.function.arguments or {}
        logger.info(f"Executing tool: {tool_name} with args: {arguments}")

        try:
            tool_result = await _execute_single_tool(agent, tool_name, arguments)

            # Cache structured data if present
            if tool_data_cache and isinstance(tool_result, dict):
                # Look for common data fields, otherwise cache the whole result
                data = tool_result.get("data") or tool_result.get("results") or tool_result
                tool_data_cache.add(tool_name, data)
                logger.debug(f"Cached tool data for {tool_name}")

            messages.append({
                "role": "tool",
                "content": str(tool_result),
                "tool_call_id": getattr(tool_call, "id", None),
            })
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {e}"
            logger.error(error_msg, exc_info=True)
            messages.append({
                "role": "tool",
                "content": error_msg,
                "tool_call_id": getattr(tool_call, "id", None),
            })

    return messages


async def _execute_single_tool(agent, tool_name: str, arguments: dict) -> Any:
    """Execute a single tool call.

    Routing priority:
    1. Agent methods (@function_tool decorated)
    2. n8n workflows (webhook-based execution)
    3. MCP servers (with server_name__tool_name prefix parsing)
    """

    # Check if it's an agent method
    if hasattr(agent, tool_name) and callable(getattr(agent, tool_name)):
        logger.info(f"Calling agent tool: {tool_name}")
        result = await getattr(agent, tool_name)(**arguments)
        logger.info(f"Agent tool {tool_name} completed")
        return result

    # Check if it's an n8n workflow
    if (
        hasattr(agent, "_n8n_workflow_name_map")
        and tool_name in agent._n8n_workflow_name_map
        and hasattr(agent, "_n8n_base_url")
        and agent._n8n_base_url
    ):
        logger.info(f"Calling n8n workflow: {tool_name}")
        workflow_name = agent._n8n_workflow_name_map[tool_name]
        result = await execute_n8n_workflow(agent._n8n_base_url, workflow_name, arguments)
        logger.info(f"n8n workflow {tool_name} completed")
        return result

    # Check MCP servers (with multi-server routing)
    if hasattr(agent, "_caal_mcp_servers") and agent._caal_mcp_servers:
        # Parse server name from prefixed tool name
        # Format: server_name__actual_tool (double underscore separator)
        if "__" in tool_name:
            server_name, actual_tool = tool_name.split("__", 1)
        else:
            # Unprefixed tools default to n8n server
            server_name, actual_tool = "n8n", tool_name

        if server_name in agent._caal_mcp_servers:
            server = agent._caal_mcp_servers[server_name]
            result = await _call_mcp_tool(server, actual_tool, arguments)
            if result is not None:
                return result

    raise ValueError(f"Tool {tool_name} not found")


async def _call_mcp_tool(mcp_server, tool_name: str, arguments: dict) -> Any | None:
    """Call a tool on an MCP server.

    Calls the tool directly without checking if it exists first - the MCP
    server will return an error if the tool doesn't exist.
    """
    if not mcp_server or not hasattr(mcp_server, "_client"):
        return None

    try:
        logger.info(f"Calling MCP tool: {tool_name}")
        result = await mcp_server._client.call_tool(tool_name, arguments)

        # Check for errors
        if result.isError:
            text_contents = []
            for content in result.content:
                if hasattr(content, "text") and content.text:
                    text_contents.append(content.text)
            error_msg = f"MCP tool {tool_name} error: {text_contents}"
            logger.error(error_msg)
            return error_msg

        # Extract text content
        text_contents = []
        for content in result.content:
            if hasattr(content, "text") and content.text:
                text_contents.append(content.text)

        return "\n".join(text_contents) if text_contents else "Tool executed successfully"

    except Exception as e:
        logger.warning(f"Error calling MCP tool {tool_name}: {e}")

    return None
