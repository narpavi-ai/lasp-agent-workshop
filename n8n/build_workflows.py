#!/usr/bin/env python3
"""Generate the importable n8n workflows that mirror the Colab labs.

    python3 n8n/build_workflows.py
    LASP_MCP_URL=http://host.docker.internal:8080/mcp python3 n8n/build_workflows.py   # local testing

Node types and typeVersions were read from the n8n source at tag n8n@2.41.6
(the version on n8n.narpavi.ca) on 2026-10-19. Versions shared with CCTP 481
use the ones 481 verified live. Re-check them before each offering.

Two things worth knowing, both from the n8n source:
- The MCP Client node (`mcpClient`) calls ONE tool and parses its JSON text,
  and it turns images into binary data, so the chain's plot shows on the canvas.
- The MCP Client Tool node (`mcpClientTool`) gives an agent the tools. It
  passes image results to the model as base64 text, so the agent excludes the
  two plot tools.
"""
import json
import os
import re
import uuid
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "..", "workshop.json")))
MCP_URL = os.environ.get("LASP_MCP_URL", CFG["mcp_url"])

PKG = "@n8n/n8n-nodes-langchain"
T_AGENT = (f"{PKG}.agent", 3.1)
T_CHAIN = (f"{PKG}.chainLlm", 1.9)
T_GEMINI = (f"{PKG}.lmChatGoogleGemini", 1.1)
T_ANTHROPIC = (f"{PKG}.lmChatAnthropic", 1.6)
T_MEMORY = (f"{PKG}.memoryBufferWindow", 1.4)
T_CHATTRIGGER = (f"{PKG}.chatTrigger", 1.4)
T_MCP = (f"{PKG}.mcpClient", 1.1)            # one tool call, a step in a chain
T_MCPTOOL = (f"{PKG}.mcpClientTool", 1.4)    # all tools, for an agent
T_MANUAL = ("n8n-nodes-base.manualTrigger", 1)
T_CODE = ("n8n-nodes-base.code", 2)
T_LOOP = ("n8n-nodes-base.splitInBatches", 3)
T_WAIT = ("n8n-nodes-base.wait", 1.1)


# Credentials are per-instance secrets and can't ship in a file. These empty
# stubs make the nodes show an unfilled credential slot. See SETUP.md.
# One folder per provider; the workflows are identical apart from the model sub-node.
PROVIDERS = {
    "gemini": {"name": "Google Gemini Chat Model", "type": T_GEMINI,
               "params": {"modelName": f"models/{CFG['models']['gemini']}", "options": {}},
               "credentials": {"googlePalmApi": {"id": "", "name": "Google Gemini(PaLM) Api account"}}},
    "claude": {"name": "Anthropic Chat Model", "type": T_ANTHROPIC,
               # Thinking off. With adaptive thinking, the n8n AI Agent replays earlier turns
               # as "[thinking][tool_use]": the model re-ran the same fits four times, then said
               # it had never sent the data (execution 27). Off, it fits once and runs IAST
               # (execution 28). 16k leaves room for long answers.
               "params": {"model": {"__rl": True, "mode": "id", "value": CFG["models"]["claude"]},
                          "options": {"thinkingMode": "disabled", "maxTokensToSample": 16000}},
               "credentials": {"anthropicApi": {"id": "", "name": "Anthropic account"}}},
}
PROVIDER = "gemini"     # set per folder by main()
MCP_CRED = {"httpBearerAuth": {"id": "", "name": "LASP MCP token"}}

# The same data, question and playbook as the notebooks.
P = [0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 5, 7, 10]
CO2 = [0.147171, 0.399272, 1.02996, 1.9223, 2.88179, 3.25169, 3.86714, 4.16431, 4.24648,
       4.41152, 4.66168, 4.89841, 5.10685, 5.2541, 5.17341, 5.23311]
N2 = [0.000319116, 0.000968288, 0.00320824, 0.00947269, 0.0311209, 0.0631523, 0.122319,
      0.186175, 0.241094, 0.29566, 0.436553, 0.552997, 0.78373, 1.14108, 1.43167, 1.77024]
QUESTION = ("Here are pure-component CO2 and N2 isotherms at 298 K on a 13X-like adsorbent.\n"
            f"pressure (bar): {P}\nCO2 loading (mol/kg): {CO2}\nN2 loading (mol/kg): {N2}\n\n"
            "Using IAST, what is the CO2/N2 selectivity for a flue gas with 15% CO2 at 1 bar total pressure?")

SYSTEM = ("You are a research assistant at the Laboratory for Advanced Separation Processes (LASP), "
          "University of Alberta, with the LASP Isotherm Fitter and IAST tools. "
          "Never estimate a number a tool can compute. Report the models you used and any "
          "workshop_validation warnings.")


def playbook():
    """The adsorbent_screening prompt: a copy of the server's, kept in adsorbent_screening.md."""
    text = open(os.path.join(HERE, "adsorbent_screening.md")).read()
    text = re.sub(r"<!--.*?-->\n", "", text, flags=re.S).strip()
    for k, v in {"gas1": "CO2", "gas2": "N2", "y1": "0.15", "pressure_bar": "1"}.items():
        text = text.replace("${" + k + "}", v)
    return text


SKILLED = SYSTEM.replace(" Report the models you used and any workshop_validation warnings.", "") + \
    "\n\nFollow this procedure:\n" + playbook()


# ------------------------------------------------------------- node helpers

def node(name, type_pair, position, parameters=None, extra=None, lab=""):
    type_name, version = type_pair
    n = {"parameters": parameters or {}, "type": type_name, "typeVersion": version,
         "position": list(position), "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"lasp/{lab}/{name}")), "name": name}
    if extra:
        n.update(extra)
    return n


def llm(position, lab, name=None):
    p = PROVIDERS[PROVIDER]
    return node(name or p["name"], p["type"], position, p["params"], {"credentials": p["credentials"]}, f"{PROVIDER}/{lab}")


def LLM():
    return PROVIDERS[PROVIDER]["name"]


def mcp_call(name, tool, json_expr, position, lab):
    return node(name, T_MCP, position, {
        "serverTransport": "httpStreamable",
        "endpointUrl": MCP_URL,
        "authentication": "bearerAuth",
        "tool": {"__rl": True, "mode": "id", "value": tool},
        "inputMode": "json",
        "jsonInput": json_expr,
        "options": {},
    }, {"credentials": MCP_CRED}, lab)


def mcp_tools(position, lab):
    return node("LASP tools (MCP)", T_MCPTOOL, position, {
        "endpointUrl": MCP_URL,
        "serverTransport": "httpStreamable",
        "authentication": "bearerAuth",
        "include": "except",
        "excludeTools": ["plot_isotherm_fit", "plot_iast"],
        "options": {},
    }, {"credentials": MCP_CRED}, lab)


def chat_trigger(position, lab):
    return node("When chat message received", T_CHATTRIGGER, position, {"options": {}},
                {"webhookId": str(uuid.uuid5(uuid.NAMESPACE_URL, f"lasp/{lab}/webhook"))}, lab)


def sticky(content, position, lab, size=(520, 300)):
    return {"parameters": {"content": content, "height": size[1], "width": size[0]},
            "type": "n8n-nodes-base.stickyNote", "typeVersion": 1, "position": list(position),
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"lasp/{lab}/sticky/{content[:40]}")),
            "name": f"Note {lab} {zlib.crc32(content[:40].encode()) % 1000}"}   # stable across runs, unlike hash()


def wire(c, src, dst, kind="main", out=0):
    branches = c.setdefault(src, {}).setdefault(kind, [])
    while len(branches) <= out:
        branches.append([])
    branches[out].append({"node": dst, "type": kind, "index": 0})


def workflow(name, nodes, connections):
    return {"name": name, "nodes": nodes, "connections": connections, "pinData": {},
            "settings": {"executionOrder": "v1"}, "meta": {"templateCredsSetupCompleted": False}}


# -------------------------------------------------------------------- labs

def lab1():
    lab, c = "01", {}
    nodes = [
        chat_trigger((0, 0), lab),
        node("Basic LLM Chain", T_CHAIN, (260, 0), {"options": {}}, lab=lab),
        llm((220, 220), lab),
        sticky("## Lab 1 · Ask the model\n\nThe twin of `01-plain-model.ipynb`. No tools.\n\n"
               "Click **Chat** and paste the question from `n8n/questions.md` (your data plus: *what is the "
               "IAST CO2/N2 selectivity at 15% CO2, 1 bar?*). Ask it twice.\n\n"
               "The LASP engines give **S ≈ 388–420**, depending on which tied N2 model you pick. "
               "What did the model say, and where did its number come from? "
               "(In our test runs: 25.6, ≈ 115, ≈ 390, ≈ 480.)\n\n"
               "**Read answers in the Chat panel**, which renders them. The node's Output table "
               "shows the raw text, with `**` and `\\n`.", (-40, -380), lab, (520, 330)),
    ]
    wire(c, "When chat message received", "Basic LLM Chain")
    wire(c, LLM(), "Basic LLM Chain", "ai_languageModel")
    return "01-plain-model.json", workflow("LASP Lab 1 · Ask the model", nodes, c)


def lab2a():
    lab, c = "02a", {}
    data_js = (f"// The same synthetic 13X-like data as the notebooks (not measured), 298 K.\n"
               f"return [{{ json: {{\n  pressure_bar: {P},\n  co2_mol_per_kg: {CO2},\n  n2_mol_per_kg: {N2},\n}} }}];")
    pick_js = ("// The fixed rule: each gas's highest-ranked fit that IAST can use.\n"
               "const top = fit => fit.ranked.find(f => f.iast_handoff);\n"
               "const co2 = top($('Fit CO2').first().json.content[0].text);\n"
               "const n2 = top($('Fit N2').first().json.content[0].text);\n"
               "return [{ json: { co2_model: co2.model, n2_model: n2.model,\n"
               "  component1: co2.iast_handoff[0], component2: n2.iast_handoff[0] } }];")
    data = "$('Workshop data').item.json"
    report = ("={{ 'Write a 3-sentence screening note for a research group from these LASP results: ' + "
              "JSON.stringify({ co2_model: $('Pick the top-ranked fits').item.json.co2_model, "
              "n2_model: $('Pick the top-ranked fits').item.json.n2_model, "
              "selectivity: $('IAST').item.json.content[0].text.points[0].selectivity_S12, "
              "henry_selectivity: $('IAST').item.json.content[0].text.henry_selectivity.value, "
              "warnings: $('IAST').item.json.content[0].text.points[0].workshop_validation.checks.filter(c => c.status !== 'pass') }) }}")
    comps = "component1: $('Pick the top-ranked fits').item.json.component1, component2: $('Pick the top-ranked fits').item.json.component2"
    nodes = [
        node("Run the chain", T_MANUAL, (0, 0), lab=lab),
        node("Workshop data", T_CODE, (220, 0), {"jsCode": data_js}, lab=lab),
        mcp_call("Fit CO2", "fit_isotherm", f"={{{{ {{ pressure: {data}.pressure_bar, loading: {data}.co2_mol_per_kg }} }}}}", (440, 0), lab),
        mcp_call("Fit N2", "fit_isotherm", f"={{{{ {{ pressure: {data}.pressure_bar, loading: {data}.n2_mol_per_kg }} }}}}", (660, 0), lab),
        node("Pick the top-ranked fits", T_CODE, (880, 0), {"jsCode": pick_js}, lab=lab),
        mcp_call("IAST", "iast_binary", f"={{{{ {{ {comps}, total_pressure_bar: [1], y1: [0.15] }} }}}}", (1100, 0), lab),
        mcp_call("Plot loadings vs P", "plot_iast", f"={{{{ {{ {comps}, panel: 'loadings_vs_P', y1: 0.15 }} }}}}", (1320, 0), lab),
        node("Write the note", T_CHAIN, (1540, 0), {"promptType": "define", "text": report, "options": {}}, lab=lab),
        llm((1500, 220), lab),
        sticky("## Lab 2a · A chain\n\nThe twin of the notebook's chain. **You** decide the steps: each node is "
               "one MCP tool call, in a fixed order. Click **Execute workflow**.\n\n"
               "- Click each node to see what the tool returned. **Plot loadings vs P** has the plot under *Binary*.\n"
               "- **Pick the top-ranked fits** applies a fixed rule. It picks Toth for N2 (S ≈ 414), but Toth, "
               "Quadratic and Langmuir are tied (S = 414, 420, 388). Nothing in the chain can notice that.\n"
               "- Only **Write the note** uses the model.", (180, -380), lab, (620, 330)),
    ]
    order = ["Run the chain", "Workshop data", "Fit CO2", "Fit N2", "Pick the top-ranked fits", "IAST",
             "Plot loadings vs P", "Write the note"]
    for a, b in zip(order, order[1:]):
        wire(c, a, b)
    wire(c, LLM(), "Write the note", "ai_languageModel")
    return "02a-chain.json", workflow("LASP Lab 2a · A chain", nodes, c)


def agent_workflow(lab, title, system, note):
    c = {}
    nodes = [
        chat_trigger((0, 0), lab),
        node("AI Agent", T_AGENT, (260, 0), {"options": {"systemMessage": system, "maxIterations": 12}}, lab=lab),
        llm((140, 240), lab),
        node("Simple Memory", T_MEMORY, (300, 240), {}, lab=lab),
        mcp_tools((460, 240), lab),
        sticky(note, (-40, -380), lab, (600, 340)),
    ]
    wire(c, "When chat message received", "AI Agent")
    wire(c, LLM(), "AI Agent", "ai_languageModel")
    wire(c, "Simple Memory", "AI Agent", "ai_memory")
    wire(c, "LASP tools (MCP)", "AI Agent", "ai_tool")
    return workflow(title, nodes, c)


def lab2b():
    return "02b-agent.json", agent_workflow("02b", "LASP Lab 2b · An agent", SYSTEM,
        "## Lab 2b · An agent\n\nThe twin of the notebook's agent. **The model** decides the steps: the AI Agent "
        "loops model → tool → model until it answers.\n\n"
        "1. Click **Chat** and ask: *What tools do you have access to?* It reads them from the MCP server.\n"
        "2. Paste the Lab 1 question (in `n8n/questions.md`). Open the AI Agent's **Logs** to see each tool call.\n"
        "3. Follow up: *Does the choice of N2 model change the selectivity? Compare the top three.* "
        "The chain (2a) can't do this without new nodes.\n\n"
        "**Simple Memory** is what lets step 3 refer to step 2: without it, every message starts from nothing, "
        "like Lab 1. **LASP tools (MCP)** leaves out the two plot tools (images would reach the model as "
        "base64 text). **Max Iterations** (12) is the safety belt.")


def lab3():
    return "03-skill.json", agent_workflow("03", "LASP Lab 3 · A skill", SKILLED,
        "## Lab 3 · A skill\n\nThe same agent, with the server's `adsorbent_screening` playbook added to its "
        "**system message**. In the notebook, the agent fetches it from the MCP server as a prompt; n8n's "
        "MCP nodes don't read prompts, so here it's pasted in (copied from the server's own text).\n\n"
        "Click **Chat** and ask: *Screen the bundled workshop_co2_298K_synthetic and workshop_n2_298K_synthetic "
        "datasets for CO2/N2 separation at 15% CO2 and 1 bar.* (The same data as Lab 2.) Then run the same "
        "question in Lab 2b and compare. The playbook has a **call plan**, a **rule for choosing** among tied "
        "fits (it picks Langmuir for N2, S ≈ 388, range 388–420) and an **output format**. In the Logs, count "
        "the tool calls. Is the answer in the format? Is the extrapolation warning there?")


SCREEN = ("Screen the bundled workshop_co2_298K_synthetic and workshop_n2_298K_synthetic datasets "
          "for CO2/N2 separation at 15% CO2 and 1 bar.")

# Facts computed by LASP's engines for SCREEN (the notebook's Lab 4 computes them live, in Step 1).
REFERENCE = ("Facts computed by LASP's solver for this question:\n"
             "- CO2 is best fitted by dsl. For N2, toth, quad, langmuir fit about equally well.\n"
             "- IAST selectivity at 1 bar and 15% CO2, by N2 model: toth 414, quad 420, langmuir 388.\n"
             "- IAST needs the N2 isotherm at about 59 bar; N2 was fitted only up to 10 bar.")
CRITERIA = {
    "selectivity": "Gives a selectivity within the reference values and says which isotherm models it used.",
    "extrapolation": "Says IAST uses the N2 isotherm far beyond its fitted range, so the result is uncertain.",
    "tied models": "Notes that several N2 models fit about equally well, and compares their selectivities "
                   "or justifies the one it chose.",
}


def lab4():
    lab, c = "04", {}
    setup_js = ("// The question, the solver's reference facts, and the criteria the judge scores against.\n"
                f"return [{{ json: {json.dumps({'task': SCREEN, 'reference': REFERENCE, 'criteria': CRITERIA}, indent=2)} }}];")
    split_js = ("// One item per criterion, each carrying the agent's answer, so the judge scores them one at a time.\n"
                "const setup = $('Question and criteria').first().json;\n"
                "const answer = $('AI Agent').first().json.output;\n"
                "return Object.entries(setup.criteria).map(([name, criterion]) =>\n"
                "  ({ json: { name, criterion, answer, reference: setup.reference } }));")
    scorecard_js = r"""// Read each verdict (tolerating text around the JSON) and build the scorecard.
const criteria = $('One item per criterion').all().map(i => i.json.name);
const rows = $input.all().map((item, i) => {
  const text = String(item.json.text ?? '');
  let score = 0, reason = 'unreadable verdict: ' + text.slice(0, 80);
  try {
    const v = JSON.parse(text.match(/\{[\s\S]*\}/)[0]);
    score = Math.min(1, Math.max(0, Number(v.score) || 0));
    reason = v.reason || '';
  } catch (e) {}
  return { criterion: criteria[i], score, reason };
});
const average = rows.reduce((a, r) => a + r.score, 0) / rows.length;
return [{ json: { average: Math.round(100 * average) / 100, rows } }];"""
    judge_prompt = ("={{ \"You are grading an AI research assistant's answer.\\n\\n\" + $json.reference + "
                    "\"\\n\\nCriterion: \" + $json.criterion + \"\\n\\nAnswer to grade:\\n\" + $json.answer + "
                    "\"\\n\\nScore from 0.0 (criterion not met) to 1.0 (fully met). "
                    "Reply with JSON only: {\\\"score\\\": <number>, \\\"reason\\\": \\\"<one sentence>\\\"}\" }}")
    nodes = [
        node("Run the evaluation", T_MANUAL, (0, 0), lab=lab),
        node("Question and criteria", T_CODE, (220, 0), {"jsCode": setup_js}, lab=lab),
        node("AI Agent", T_AGENT, (460, 0), {"promptType": "define", "text": "={{ $json.task }}",
             "options": {"systemMessage": SKILLED, "maxIterations": 12}}, lab=lab),
        llm((380, 240), lab),
        mcp_tools((560, 240), lab),
        node("One item per criterion", T_CODE, (760, 0), {"jsCode": split_js}, lab=lab),
        node("LLM judge", T_CHAIN, (980, 0), {"promptType": "define", "text": judge_prompt, "options": {}}, lab=lab),
        llm((960, 240), lab, name="Judge model"),
        node("Scorecard", T_CODE, (1200, 0), {"jsCode": scorecard_js}, lab=lab),
        sticky("## Lab 4 · Evaluation (LLM-as-judge)\n\nThe twin of `04-evaluation.ipynb`. The skilled agent answers "
               "Lab 3's question once; then a second model (**Judge model**) scores that answer on three criteria, "
               "using **reference facts from LASP's solver** (in **Question and criteria**).\n\n"
               "Click **Execute workflow**, then open **Scorecard**: a score and a reason per criterion.\n\n"
               "**Compare:** delete the procedure from the AI Agent's system message (the plain agent) and run it "
               "again. Did *extrapolation* and *tied models* drop? Then read the reasons: do you agree with the judge?",
               (-40, -420), lab, (620, 340)),
    ]
    order = ["Run the evaluation", "Question and criteria", "AI Agent", "One item per criterion", "LLM judge", "Scorecard"]
    for a, b in zip(order, order[1:]):
        wire(c, a, b)
    wire(c, LLM(), "AI Agent", "ai_languageModel")
    wire(c, "Judge model", "LLM judge", "ai_languageModel")
    wire(c, "LASP tools (MCP)", "AI Agent", "ai_tool")
    return "04-evaluation.json", workflow("LASP Lab 4 · Evaluation (LLM-as-judge)", nodes, c)


def main():
    global PROVIDER
    for PROVIDER in PROVIDERS:
        os.makedirs(os.path.join(HERE, PROVIDER), exist_ok=True)
        for old in ("03a-skill.json", "03b-evaluation.json"):
            if os.path.exists(os.path.join(HERE, PROVIDER, old)):
                os.remove(os.path.join(HERE, PROVIDER, old))
        for build in (lab1, lab2a, lab2b, lab3, lab4):
            name, wf = build()
            with open(os.path.join(HERE, PROVIDER, name), "w") as f:
                json.dump(wf, f, indent=2, ensure_ascii=False)
                f.write("\n")
        print(f"wrote n8n/{PROVIDER}/  (5 workflows)")
    with open(os.path.join(HERE, "questions.md"), "w") as f:
        f.write("# Questions to paste into the chat\n\n"
                "## Lab 2b, first\n\n```\nWhat tools do you have access to?\n```\n\n"
                "## Labs 1 and 2b: the screening question\n\n```\n" + QUESTION + "\n```\n\n"
                "## Lab 2b follow-up\n\n```\nDoes the choice of N2 model change the selectivity? "
                "Compare the top three IAST-compatible N2 fits.\n```\n\n"
                "## Lab 3 (and the same in Lab 2b, to compare)\n\n```\n" + SCREEN + "\n```\n")
    print("wrote n8n/questions.md")


if __name__ == "__main__":
    main()
