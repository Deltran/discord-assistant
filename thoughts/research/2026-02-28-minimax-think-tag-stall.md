# Research: MiniMax `<think>` Tag Stall — Why the Bot Stops After Thinking

**Date**: 2026-02-28
**Git Commit**: b698642bea0404a145df7ad06a5e7da1dfcd287d
**Branch**: main

## Research Question

The bot frequently outputs a `<think>` block and then stalls — it does not continue with a visible response. Investigate: how does the bot call MiniMax, how is the response handled, does it strip thinking tags, is there streaming, and are there any timeout/max_token limits that could cause a mid-response stop?

## Summary

The bot calls MiniMax via `langchain_openai.ChatOpenAI` using a non-streaming `ainvoke()` call. MiniMax M2.5 returns `<think>...</think>` blocks inline in the response `content` string. The code in `tool_loop.py` strips those blocks via regex before returning the text. **The stall is not caused by a missing strip — the strip is implemented.** The root cause is a sequencing problem: `_sanitize_minimax_response` is only called inside `run_tool_loop`, but when the agent has no tools bound, `CoreAgent.invoke` calls `self.llm.ainvoke()` directly and skips sanitization entirely, returning raw content including any `<think>...</think>` blocks verbatim. Additionally, when tools ARE bound, if MiniMax returns a response whose content is _only_ a think block (empty after stripping), the code returns an empty string, which the bot silently drops.

There is no streaming, no timeout setting, and no max_tokens cap anywhere in the codebase.

## Detailed Findings

### 1. LLM Construction — `src/providers/minimax.py`

`create_llm()` at line 65 creates a bare `ChatOpenAI` instance:

```python
ChatOpenAI(
    model=settings.minimax_model,   # "MiniMax-M2.5"
    api_key=settings.minimax_api_key.get_secret_value(),
    base_url=settings.minimax_base_url,  # "https://api.minimax.io/v1"
    temperature=0.7,
)
```

- No `streaming=True`
- No `max_tokens`
- No `timeout`
- No `request_timeout`
- No stop sequences

All calls are standard blocking (async) completions. The response arrives in one piece.

### 2. Two Distinct Call Paths — `src/agent/core.py:158–167`

`CoreAgent.invoke()` branches on whether tools exist:

```python
if self.tools:
    response = await run_tool_loop(...)
else:
    response = await self.llm.ainvoke(normalized)
```

- **With tools** (the live configuration — tools are always bound in `main.py`): goes through `run_tool_loop` in `tool_loop.py`, which includes `_sanitize_minimax_response`.
- **Without tools**: `ainvoke()` is called directly, and the raw response including `<think>` blocks is returned unchanged to the caller. This path is currently unused in production but represents a latent bug.

In production, `main.py:60–72` always constructs a tools list and passes it to `CoreAgent`, so the `run_tool_loop` path is always taken.

### 3. Think-Tag Stripping — `src/agent/tool_loop.py:15–29`

The sanitizer is defined at the top of `tool_loop.py`:

```python
def _sanitize_minimax_response(content: str) -> tuple[str, bool]:
    had_tool_calls = bool(
        re.search(r"<minimax:tool_call>.*?</minimax:tool_call>", content, re.DOTALL)
    )
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    cleaned = re.sub(
        r"<minimax:tool_call>.*?</minimax:tool_call>", "", cleaned, flags=re.DOTALL
    )
    return cleaned.strip(), had_tool_calls
```

This regex is **greedy-off** (`.*?` = non-greedy) and uses `re.DOTALL`, so it will correctly match think blocks that span multiple lines. It handles the case where MiniMax emits only a think block and no tool call XML by returning an empty string after `.strip()`.

### 4. Where Sanitization Is Applied in the Tool Loop — `src/agent/tool_loop.py:59–78`

```python
if not response.tool_calls:
    content = response.content or ""
    cleaned, had_tool_calls = _sanitize_minimax_response(content)

    if had_tool_calls:
        # retry path — injects a correction message and loops
        ...
        continue

    if cleaned != content:
        # only think blocks were stripped — return cleaned content
        return AIMessage(content=cleaned)

    return response
```

The sanitizer is only invoked on the **non-tool-call branch**, i.e., when MiniMax returns a final text answer (no function calls). If `cleaned` is empty (think block only, no actual text), the code returns `AIMessage(content="")`.

Back in `client.py:104`, the bot checks `if response:` before sending. An empty string is falsy, so an empty response is **silently dropped** — the bot says nothing at all.

### 5. The Stall Scenario — Most Likely Root Cause

When MiniMax M2.5 emits a `<think>` block followed by actual response text in the same completion, the flow works correctly — the think block is stripped and the text is returned.

The stall happens when MiniMax returns **only a `<think>` block with no following text** in that API call. This can occur because:

1. MiniMax M2.5 may emit thinking as a "first token" response and then require another generation turn to produce the actual answer. If the API call completes after the think block with no further content, `cleaned` is `""`.
2. `AIMessage(content="")` is returned, the bot sees a falsy response, sends nothing.

From the user's perspective: the bot appeared to "think" (possibly a progress message or partial Discord content), then went silent.

A secondary scenario: if MiniMax uses **extended thinking** where the think tag is never closed (`<think>...` without `</think>`), the non-greedy `.*?` regex will not match it (because it requires the closing tag), and the raw `<think>` text is returned verbatim to Discord. This would look like the bot outputting the `<think>` text directly.

### 6. Progress Indicator — `src/bot/client.py:83–85`

```python
async with message.channel.typing():
    response = await self._agent_callback(message)
```

The bot holds the typing indicator for the entire duration of the LLM call. From the user's perspective, the typing indicator shows, the call completes with an empty response, the typing indicator disappears, and nothing is sent. This matches the "stall" description.

### 7. No Streaming

Confirmed: there is no `stream=True`, no `astream()` call, and no token-by-token Discord message editing anywhere in the codebase. The full response is awaited before anything is sent to Discord. The think output the user sees is not live-streamed content — it must be coming from somewhere else (possibly a prior partial response, a different code path, or a logging artifact).

### 8. No Timeout or Max Tokens Config

Searched across the entire codebase:
- No `request_timeout`, `timeout`, or `max_retries` in `create_llm()`
- No `max_tokens` anywhere in provider or agent code
- The only iteration cap is `max_iterations=10` in `run_tool_loop`, which limits tool-call loops, not token generation

### 9. Think-Tag Handling in Session History — `src/agent/core.py:198`

```python
ai_msg = AIMessage(content=response.content)
session.append(ai_msg)
```

`response.content` here is already the cleaned content (returned from `run_tool_loop`). So think blocks do not accumulate in session history. However, if `response.content` is `""` (the stall scenario), an empty AIMessage is appended to the session. On the next turn, `normalize_messages` may merge this with surrounding messages or pass it through — this has not been traced further but represents a potential session corruption vector.

## Code References

- `src/providers/minimax.py:65-72` — `create_llm()`: no streaming, no timeout, no max_tokens
- `src/providers/minimax.py:9-62` — `normalize_messages()`: enforces strict user/assistant alternation for MiniMax
- `src/agent/core.py:158-167` — branching between `run_tool_loop` and bare `ainvoke`
- `src/agent/core.py:198` — session append uses already-cleaned `response.content`
- `src/agent/tool_loop.py:15-29` — `_sanitize_minimax_response()`: regex strips `<think>` and `<minimax:tool_call>` blocks
- `src/agent/tool_loop.py:55-78` — tool loop exit path: sanitizer called, empty string returned if think-only response
- `src/bot/client.py:83-106` — typing indicator held during full `agent_callback` await; `if response:` guard silently drops empty string
- `src/settings.py:9-35` — Settings: no timeout, max_tokens, or streaming config fields

## Architecture Notes

- MiniMax M2.5 is accessed through `langchain_openai.ChatOpenAI` using MiniMax's OpenAI-compatible endpoint. MiniMax does not use standard OpenAI function-calling format — it sometimes emits tool calls as `<minimax:tool_call>` XML in the content string rather than in the structured `tool_calls` field. The code handles this via the sanitizer's retry loop.
- The `normalize_messages` layer exists because MiniMax requires strict alternating user/assistant turns and rejects consecutive same-role messages.
- The session history is kept in-memory per process (Python dict in `CoreAgent._sessions`). No persistence across restarts for session content (only the SQLite message log and vector store persist).

## Open Questions

1. **Is MiniMax returning an unclosed `<think>` tag?** If `</think>` is absent, the regex fails to strip and the raw tag appears in Discord. Logs would confirm this.
2. **Is the think block the entire completion, or is text following it?** If MiniMax returns `<think>...</think>` and nothing after, the empty-response path is the stall mechanism.
3. **Does MiniMax's API return thinking in a separate field** (like Anthropic's `thinking` content blocks) rather than inline in `content`? If so, LangChain may be surfacing it in `content` due to the OpenAI-compat shim, and the underlying behavior may differ from what the regex expects.
4. **What does the raw API response JSON look like?** Adding a debug log of `response.content` before sanitization would confirm which scenario is occurring.
