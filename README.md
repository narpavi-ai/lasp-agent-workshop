# Build your agent: LASP workshop

Agentic AI workshop for the Laboratory for Advanced Separation Processes (LASP),
University of Alberta.

The running example: **screen an adsorbent for CO₂/N₂ separation** with the
group's own [Isotherm Fitter and IAST apps](https://lasp.engineering.ualberta.ca/apps.html),
which an agent calls as tools through an MCP server. The server runs the apps'
engines unchanged on Cloud Run; it lives in a separate private repo,
`narpavi-ai/lasp-mcp-server`. The labs only need its URL and the workshop token.

| | |
|---|---|
| [`notebooks/`](notebooks/) | Colab track: Python and LangGraph, with Gemini (free) or Claude |
| [`n8n/`](n8n/) | No-code track: the same labs as n8n workflows, in `gemini/` and `claude/` folders |
| [`docs/`](docs/) | [Before the workshop](docs/before-the-workshop.md) · [Colab ↔ n8n mapping](docs/concept-mapping.md) |

Every lab exists in both tracks, and both build the same agent against the same server.

| Lab | What it shows |
|---|---|
| 1. Plain model | Asked for the selectivity, the model **guesses**. |
| 2. MCP and an agent | The LASP tools over MCP: first you run the steps yourself, then an agent picks its own |
| 3. Skill | A screening playbook from the server, and what it changes |
| 4. Evaluation | An LLM judge, given the solver's facts, scores both agents: did the skill help? |

`workshop.json` holds the server URL and the models, shared by both tracks.

---

*Generated with AI assistance as part of workshop prep; may contain errors.
Verify against the linked official docs before relying on anything technical.*
