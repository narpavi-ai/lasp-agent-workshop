"""Helpers for the LASP workshop notebooks: secrets, display, the MCP connection, traces.

Each notebook downloads this file from the workshop repo and imports it, so the
notebooks show only what's being taught. Nothing here is needed to understand
the labs; read it if you're curious.
"""
import base64
import getpass
import json
import logging
import os

from IPython.display import HTML, Image, Markdown, display

# Gemini's converter prints a warning for every tool because our tool schemas
# carry a standard "$schema" line it doesn't use. Harmless, so keep it quiet.
logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)

__all__ = ["secret", "show", "show_text", "text_of", "connect", "call", "show_fit", "show_iast",
           "run_live", "answer_of", "tools_used", "draw"]

TOOLS = {}


# ----------------------------------------------------------------- secrets

def secret(name, prompt=None):
    """Colab Secrets first; otherwise an environment variable; otherwise ask
    (hidden input, never saved in the notebook). Spaces and newlines are removed."""
    value = None
    try:
        from google.colab import userdata
        value = userdata.get(name)
    except Exception:
        value = os.environ.get(name)
    if not value:
        value = getpass.getpass(prompt or f"{name}: ")
    return value.strip()


# ----------------------------------------------------------------- display

def _text_of(message):
    """The readable text of a model message. Gemini 3 returns a list of blocks."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n\n".join(b if isinstance(b, str) else b.get("text", "")
                           for b in content if isinstance(b, str) or b.get("type") == "text")
    return str(content)


text_of = _text_of


def _card(body, label=None, icon="", limit=1500):
    body = (body if isinstance(body, str) else _text_of(body)).strip() or "*(nothing came back)*"
    if len(body) > limit:
        body = body[:limit].rstrip() + f"\n\n*… trimmed, {len(body):,} characters in full*"
    head = [f"> {icon} **{label}**", ">"] if label else []
    return "\n".join(head + ["> " + line for line in body.splitlines()])


def show(response, label="Model answer", note=None):
    """A model's answer, in full: a labelled header, a one-line caption, then the text."""
    caption = f'<div style="opacity:.75;font-size:.9em;margin-top:2px">{note}</div>' if note else ""
    display(HTML(f'<div style="border-left:4px solid #2563eb;background:rgba(37,99,235,.08);'
                 f'padding:8px 12px;margin:14px 0 6px"><b>🤖 {label}</b>{caption}</div>'))
    display(Markdown(_text_of(response).strip() or "*(nothing came back)*"))
    display(HTML('<hr style="opacity:.25;margin:10px 0 18px">'))


def show_text(text, label=None, icon="📄"):
    display(Markdown(_card(text, label, icon=icon, limit=2500)))


# ----------------------------------------------------------------- MCP

def _explain(e, url):
    """The MCP client wraps the real error in an ExceptionGroup; show the cause instead."""
    while isinstance(e, BaseExceptionGroup):
        e = e.exceptions[0]
    msg = f"{type(e).__name__}: {e}"
    hint = ("401: check the LASP_MCP_TOKEN secret: exactly 48 characters, 0-9 and a-f, nothing else." if "401" in msg else
            "The token has a space or newline in it: re-paste the LASP_MCP_TOKEN secret." if "header" in msg.lower() else
            "Can't reach the server: check the URL, and your internet connection.")
    raise RuntimeError(f"Couldn't connect to {url}\n{msg}\n→ {hint}") from None


async def connect(url, token, extra_servers=None):
    """Connect to the LASP MCP server and load its tools. Returns (client, tools)."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    servers = {"lasp": {"transport": "streamable_http", "url": url.strip(),
                        "headers": {"Authorization": f"Bearer {token.strip()}"}}}
    servers.update(extra_servers or {})
    client = MultiServerMCPClient(servers)
    try:
        tools = await client.get_tools()
    except BaseException as e:
        _explain(e, url)
    TOOLS.clear()
    TOOLS.update({t.name: t for t in tools})
    return client, tools


def _tool_text(content):
    blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
    return "\n".join(b["text"] for b in blocks if isinstance(b, dict) and b.get("type") == "text")


async def call(tool, /, **args):
    """Call a LASP tool yourself, no model involved. Shows any plot; returns the JSON.
    (`tool` is positional-only, so a tool argument called `name` still works.)"""
    blocks = await TOOLS[tool].ainvoke(args)
    blocks = blocks if isinstance(blocks, list) else [{"type": "text", "text": blocks}]
    for b in blocks:
        if b.get("type") == "image":
            display(Image(data=base64.b64decode(b["base64"])))
    text = _tool_text(blocks)
    return json.loads(text) if text else None


# ----------------------------------------------------------------- readable tool results

def _fit_table(r):
    rows = ["| rank | model | AICc | R² | workshop check |", "|---|---|---|---|---|"]
    for i, f in enumerate(r["ranked"][:5], 1):
        rows.append(f"| {i} | {f['model']} | {f['aicc']:.1f} | {f['r2']:.4f} | {f['workshop_validation']['status']} |")
    return f"{r['summary']['models_fitted']} models fitted. Top 5 by AICc:\n\n" + "\n".join(rows)


def _iast_lines(r):
    lines = [r["henry_selectivity"]["text"]]
    for p in r["points"]:
        warn = [c["check"] for c in p["workshop_validation"]["checks"] if c["status"] != "pass"]
        lines.append(f"P = {p['total_pressure_bar']} bar, y1 = {p['y1']}: **S12 = {p['selectivity_S12']:.1f}**"
                     + (f"  ⚠️ {', '.join(warn)}" if warn else ""))
    return "\n\n".join(lines)


def show_fit(result, label="fit_isotherm"):
    show_text(_fit_table(result), label, icon="🛠️")


def show_iast(result, label="iast_binary"):
    show_text(_iast_lines(result), label, icon="🛠️")


def _brief(name, text):
    try:
        r = json.loads(text)
    except (TypeError, ValueError):
        return text
    if name == "fit_isotherm":
        return _fit_table(r)
    if name == "iast_binary":
        return _iast_lines(r)
    return text


# ----------------------------------------------------------------- agent traces

def _short(v):
    """Tool arguments, abbreviated: a list of 16 numbers prints as [16 values]."""
    if isinstance(v, list) and len(v) > 6:
        return f"[{len(v)} values]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_short(x)}" for k, x in v.items()) + "}"
    return repr(v)


def _step(m):
    if type(m).__name__ == "AIMessage" and m.tool_calls:
        calls = "\n\n".join(f"`{tc['name']}({', '.join(f'{k}={_short(v)}' for k, v in tc['args'].items())})`"
                            for tc in m.tool_calls)
        return _card(calls, "Model called a tool", icon="🔧", limit=900)
    if type(m).__name__ == "ToolMessage":
        return _card(_brief(m.name, _tool_text(m.content)), f"{m.name} returned", icon="🛠️", limit=1200)
    return _card(m, type(m).__name__)


async def run_live(agent, question, recursion_limit=16, quiet=False):
    """Run an agent, showing each step as it happens. Returns every new message."""
    if not quiet:
        display(Markdown(_card(question, "You asked", icon="👤", limit=500)))
    new = []
    async for update in agent.astream({"messages": [{"role": "user", "content": question}]},
                                      {"recursion_limit": recursion_limit}, stream_mode="updates"):
        for out in update.values():
            for m in (out or {}).get("messages", []):
                new.append(m)
                if quiet:
                    continue
                if type(m).__name__ == "AIMessage" and not m.tool_calls:
                    show(m, "Agent answered", "Its final answer, after the tool calls above.")
                else:
                    display(Markdown(_step(m)))
    if not quiet:
        requests = sum(type(m).__name__ == "AIMessage" for m in new)
        calls = sum(type(m).__name__ == "ToolMessage" for m in new)
        display(Markdown(f"*{requests} model requests, {calls} tool calls.*"))
    return new


def answer_of(messages):
    """The agent's final answer, as text."""
    return _text_of(next(m for m in reversed(messages) if type(m).__name__ == "AIMessage"))


def tools_used(messages):
    """The names of the tools the agent called, in order."""
    return [tc["name"] for m in messages if type(m).__name__ == "AIMessage" for tc in m.tool_calls]


def draw(graph):
    """Draw a LangGraph graph. Uses mermaid.ink online; falls back to text."""
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        print(graph.get_graph().draw_mermaid())
