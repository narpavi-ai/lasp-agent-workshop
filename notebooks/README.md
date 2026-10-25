# Colab track

Python labs with LangGraph, calling the LASP MCP server.
Open a notebook on GitHub and click **Open in Colab**. Setup is in
[before the workshop](../docs/before-the-workshop.md).

| Lab | Notebook | What you build | Requests |
|---|---|---|---|
| 1 | [`01-plain-model.ipynb`](01-plain-model.ipynb) | A plain model call. Asked for the selectivity, it **guesses**. | ~5 |
| 2 | [`02-mcp-and-agent.ipynb`](02-mcp-and-agent.ipynb) | **What MCP is**; calling the LASP tools yourself (a workflow); then an **agent** with `create_agent`, its loop drawn | ~12 |
| 3 | [`03-skill.ipynb`](03-skill.ipynb) | The server's screening playbook as a **skill**: the same agent with and without it | ~10 |
| 4 | [`04-evaluation.ipynb`](04-evaluation.ipynb) | **LLM-as-judge**: a second model scores plain vs skilled answers on three criteria, against reference facts from the solver | ~16 |

Each lab is self-contained: every notebook is a fresh Colab runtime. Helper code
(display, connection, traces) is in [`lasp_lab.py`](lasp_lab.py); each notebook
downloads it from this repo in its first setup cell.

**Gemini or Claude:** every lab starts with `PROVIDER = "gemini"`. Change it to
`"claude"` and add an `ANTHROPIC_API_KEY` secret, which is a paid key. Nothing
else in the lab changes. Claude accepts the tools' schemas as written.

## For maintainers

The notebooks are **generated**. Edit `build_notebooks.py` (or `../workshop.json`
for the server URL and model, or `lasp_lab.py` for the helpers), then rebuild:

```bash
python3 notebooks/build_notebooks.py
```

Test against the live server (token from Secret Manager), or a local copy with
`LASP_MCP_URL=http://localhost:8080/mcp LASP_MCP_TOKEN=dev-token`:

```bash
LASP_MCP_TOKEN=... python notebooks/test_notebooks.py      # scripted stand-in model, no key
python notebooks/test_notebooks.py --live                  # real Gemini (GOOGLE_API_KEY)
LAB_PROVIDER=claude python notebooks/test_notebooks.py --live   # real Claude (ANTHROPIC_API_KEY)
```

The scripted run tests every part except Gemini's own decisions. **Do the live
run before the workshop.** Tested against `langchain` 1.4.7, `langgraph` 1.2.17,
`langchain-mcp-adapters` 0.3.2, `langchain-google-genai` 4.4.2, `langchain-anthropic` 1.7.6.
