# MiniMax Tool Call Sanitizer

**Date:** 2026-02-27
**Status:** Approved

## Problem

MiniMax M2.5 sometimes emits tool calls as raw XML text (`<minimax:tool_call>`) and thinking blocks (`<think>`) in its response content, instead of using the OpenAI-compatible function calling interface that LangChain expects. When this happens:

1. `response.tool_calls` is empty (LangChain sees no structured tool calls)
2. The tool loop exits immediately, treating the XML as a final text response
3. The raw XML gets sent to Discord as the bot's reply
4. The user sees internal monologue + broken tool invocations instead of an answer

## Fix

Add post-processing to `run_tool_loop` in `src/agent/tool_loop.py`. After receiving a response with no `tool_calls`, check if the text content contains MiniMax XML tool call patterns. If detected:

1. Strip the `<think>` block and `<minimax:tool_call>` XML from the response text
2. Append a ToolMessage-style correction telling MiniMax to use the proper function calling interface with the list of available tools
3. Continue the loop (MiniMax self-corrects on retry)

### Detection

Regex match on response content for:
- `<minimax:tool_call>` — MiniMax's XML tool call syntax
- `<think>` blocks — MiniMax's chain-of-thought leakage (strip but don't trigger retry on its own)

### Correction Message

When XML tool calls are detected, replace the AI response content with the stripped text (if any remains) and append a HumanMessage:

> "You attempted to call a tool using XML syntax, but that doesn't work. Use the function calling interface instead. Available tools: {tool_names}"

### Iteration Budget

The correction consumes one loop iteration. No special handling needed — the existing `max_iterations` cap covers it.

## Scope

- **Only file changed:** `src/agent/tool_loop.py`
- **No changes to:** provider, agent core, bot client, or tools
- **Test:** Unit test that simulates MiniMax returning XML tool call text and verifies the loop retries with correction

## Non-Goals

- Parsing the XML and executing the tool (fragile, rewards bad behavior)
- Changing MiniMax's system prompt to prevent this (defense in depth — do both eventually, but the sanitizer is the reliable fix)
- Handling `<think>` blocks separately (strip them in the same pass)
