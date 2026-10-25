# n8n track

The same labs as the [notebooks](../notebooks/), as importable workflows for the
workshop's n8n. Each canvas has a sticky note with what to try. The
[concept mapping](../docs/concept-mapping.md) puts Python and the canvas side by side.

**Pick the folder that matches your key:** [`gemini/`](gemini/) (free) or
[`claude/`](claude/) (paid Anthropic key). The workflows in the two folders are
identical apart from the model sub-node, so you can also swap that node yourself.

| Lab | Workflow | What it is |
|---|---|---|
| 1 | `01-plain-model.json` | Chat → Basic LLM Chain. No tools. |
| 2a | `02a-chain.json` | **A chain:** MCP Client nodes in a row (fit CO₂ → fit N₂ → pick → IAST → plot), then the model writes a note |
| 2b | `02b-agent.json` | **An agent:** AI Agent + MCP Client Tool + memory |
| 3 | `03-skill.json` | The same agent, with the screening playbook in its system message |
| 4 | `04-evaluation.json` | **LLM-as-judge**: the agent answers once, then a second model (**Judge model**) scores it on three criteria against the solver's facts |

The questions to paste into the chat are in [`questions.md`](questions.md).

## Setup, once

1. **Credentials → Add credential:**
   - **Google Gemini(PaLM) Api**: your key from <https://aistudio.google.com/apikey>.
     Or, for the `claude/` workflows, **Anthropic**: your key from <https://console.anthropic.com>.
   - **Bearer Auth**: name it `LASP MCP token`, and paste the workshop token.
2. **Create workflow → ⋯ → Import from URL…**, and paste the workflow's raw link:
   `https://raw.githubusercontent.com/narpavi-ai/lasp-agent-workshop/main/n8n/gemini/02b-agent.json`
   (change `gemini` and the file name as needed). Or download the file and use
   **Import from file…**.
3. On each model node and MCP node, select your credential.
4. Click **Chat** for Labs 1, 2b and 3, or **Execute workflow** for 2a and 4.

## What differs from the notebooks

- **The agent leaves out the plot tools.** The MCP Client Tool node would send
  each image to the model as base64 text. The chain's **MCP Client** node turns
  images into binary data, so its plot shows under **Binary**.
- **The skill is pasted, not fetched.** n8n's MCP nodes don't read MCP
  prompts, so the playbook's text is kept in
  [`adsorbent_screening.md`](adsorbent_screening.md), a copy of the server's.

## For maintainers

The workflows are **generated**. Edit `build_workflows.py`, or `../workshop.json`
for the server URL and model, then run:

```bash
python3 n8n/build_workflows.py && python3 n8n/validate_workflows.py
```

**Tested on n8n 2.41.6, the version on n8n.narpavi.ca:**
- All ten workflows import, from both folders.
- The chain runs end to end against the server: S = 413.9 with the
  extrapolation warning, and the plot arrives as a PNG.

The agent workflows need a live key to test, so run them before the
workshop.
