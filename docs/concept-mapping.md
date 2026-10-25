# Colab ↔ n8n: the same concepts

n8n's AI nodes are LangChain, the JavaScript version, so the concepts map one
to one even though the code doesn't.

| Concept | Colab (LangGraph, Python) | n8n |
|---|---|---|
| The model | `init_chat_model(MODELS[PROVIDER])`: Gemini or Claude | **Google Gemini Chat Model** or **Anthropic Chat Model** sub-node (`n8n/gemini/` or `n8n/claude/`) |
| A plain call | `model.invoke(...)` | **Basic LLM Chain** |
| The MCP server | `MultiServerMCPClient({... "url": ..., "headers": ...})` | an endpoint URL plus a **Bearer Auth** credential |
| One tool call | `await call("fit_isotherm", ...)` | **MCP Client** node |
| Shared data between steps | Python variables | `$('Node name').item.json` |
| A workflow (fixed steps) | `call()` the tools in order, in plain code | MCP Client nodes in a row (Lab 2a) |
| An agent | `create_agent(model, tools, system_prompt=...)`: LangGraph runs the loop | **AI Agent** node |
| The agent's tools | `model.bind_tools(agent_tools)` | **MCP Client Tool** sub-node (plot tools excluded) |
| Memory | none: each `run_live` starts fresh | **Simple Memory** sub-node, so chat follow-ups work |
| Safety belt | `recursion_limit` | **Max Iterations** |
| A skill | the server's MCP prompt, fetched with `client.get_prompt(...)` | the same text, pasted into the **System Message** |
| Evaluation (LLM-as-judge) | `judge(answer, criterion)`: a second model call with the solver's facts | **Code** (criteria) → AI Agent → **Basic LLM Chain** as judge → **Code** (scorecard) |
| Seeing the steps | `run_live()` streams each one | the AI Agent node's **Logs** |
| Seeing a plot | displayed inline from the tool result | the MCP Client node's **Binary** tab |
