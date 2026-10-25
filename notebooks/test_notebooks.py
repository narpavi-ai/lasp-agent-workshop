#!/usr/bin/env python3
"""Run every lab notebook end to end against a local server, without a Gemini key.

A scripted stand-in replaces the model: it reads the conversation and makes the
tool calls a sensible model would. That exercises everything except the
model's own judgement: setup, MCP connection, direct tool calls, the LangGraph
chain and agent, streaming traces and the evaluation scorer.

    cd ../lasp-mcp-server && npm run inspect   # private repo; server on localhost:8080, token dev-token
    python notebooks/test_notebooks.py    # needs: nbclient ipykernel + the lab packages
    python notebooks/test_notebooks.py --live   # real Gemini; needs GOOGLE_API_KEY
    LAB_PROVIDER=claude python notebooks/test_notebooks.py [--live]   # Claude; live needs ANTHROPIC_API_KEY

The live run is the one that matters before a workshop.
"""
import os
import re
import sys

import nbformat
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE = "--live" in sys.argv
PROVIDER = os.environ.get("LAB_PROVIDER", "gemini")   # gemini | claude
DEPLOYED_URL = __import__("json").load(open(os.path.join(HERE, "..", "workshop.json")))["mcp_url"]

FAKE_MODEL = r'''
# ---- injected by test_notebooks.py: a scripted stand-in for the model ----
import json, re
from langchain_core.language_models.fake_chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

def _nums(name, text):
    m = re.search(name + r"[^\[]*\[([^\]]+)\]", text)
    return [float(x) for x in m.group(1).split(",")] if m else None

class Scripted(BaseChatModel):
    calls: int = 0
    @property
    def _llm_type(self): return "scripted"
    def bind_tools(self, tools, **kw): return self
    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])
    def _next(self, messages):
        human = next(m.content for m in messages if m.type == "human")
        human = human if isinstance(human, str) else str(human)
        results = {}
        for m in messages:
            if m.type == "tool":
                text = m.content if isinstance(m.content, str) else "".join(b.get("text", "") for b in m.content if isinstance(b, dict))
                results.setdefault(m.name, []).append(json.loads(text))
        tc = lambda name, args: {"name": name, "args": args, "id": f"c{self.calls}{name}{len(str(args))}"}
        if human.startswith("You are grading"):
            return AIMessage(content='{"score": 1, "reason": "scripted judge"}')
        if not any(m.type == "ai" and m.tool_calls for m in messages) and not re.search(r"ANSWER|Fit |Screen|IAST|bundled", human):
            return AIMessage(content="A scripted answer: IAST assumes an ideal adsorbed phase.")
        sets = re.findall(r"(workshop_\w+_synthetic|lasp_example_three_T)", human)
        if sets and "get_sample_dataset" not in results:
            return AIMessage(content="", tool_calls=[tc("get_sample_dataset", {"name": s}) for s in dict.fromkeys(sets)])
        data = []
        if "get_sample_dataset" in results:
            data = [(d["pressure_bar"], d["loading_mol_per_kg"], d.get("temperature_K")) for d in results["get_sample_dataset"]]
        elif _nums("CO2 loading", human):
            data = [(_nums("pressure", human), _nums("CO2 loading", human), None), (_nums("pressure", human), _nums("N2 loading", human), None)]
        if data and "fit_isotherm" not in results:
            calls = []
            for p, q, T in data:
                args = {"pressure": p, "loading": q}
                if T: args.update(temperature_K=T, temperature_mode="global")
                calls.append(tc("fit_isotherm", args))
            return AIMessage(content="", tool_calls=calls)
        fits = results.get("fit_isotherm", [])
        h = lambda f: next(r for r in f["ranked"] if r.get("iast_handoff"))["iast_handoff"][0]
        if "ranks first" in human and fits:
            return AIMessage(content=f"ANSWER: {fits[0]['ranked'][0]['model']}")
        if "reversal" in human and len(fits) >= 2:
            if "iast_sweep_summary" not in results:
                return AIMessage(content="", tool_calls=[tc("iast_sweep_summary", {"component1": h(fits[0]), "component2": h(fits[1])})])
            return AIMessage(content="ANSWER: " + ("yes" if results["iast_sweep_summary"][0]["selectivity_reversal"]["pressures_bar"] else "no"))
        if len(fits) == 1 and fits[0]["isosteric_heat"]["from_top_ranked_global_fit"]:
            q = fits[0]["isosteric_heat"]["from_top_ranked_global_fit"]["qst_kJ_per_mol"][0]
            return AIMessage(content=f"Isosteric heat {q:.1f} kJ/mol.\nANSWER: {q:.1f}")
        if len(fits) >= 2 and "iast_binary" not in results:
            return AIMessage(content="", tool_calls=[tc("iast_binary", {"component1": h(fits[0]), "component2": h(fits[1]),
                                                                          "total_pressure_bar": [1], "y1": [0.15]})])
        if "iast_binary" in results:
            r = results["iast_binary"][0]; s = r["points"][0]["selectivity_S12"]
            if "Henry" in human: return AIMessage(content=f"ANSWER: {r['henry_selectivity']['value']:.1f}")
            if "beyond" in human: return AIMessage(content="N2 is extrapolated.\nANSWER: yes, N2")
            return AIMessage(content=f"Selectivity {s:.1f}. ANSWER: {s:.1f}")
        if fits:
            return AIMessage(content=f"ANSWER: {fits[0]['ranked'][0]['model']}")
        return AIMessage(content="ANSWER: unknown")

model = Scripted()
print("🧪 scripted model in place of the real one")
'''

FAKE_DRAW = r'''
# ---- injected: no network for mermaid.ink in CI; print the graph text instead ----
def draw(graph):
    print(graph.get_graph().draw_mermaid()[:400])
'''


def run(name):
    nb = nbformat.read(os.path.join(HERE, name), as_version=4)
    cells = []
    for c in nb.cells:
        if c.cell_type == "code" and "%pip install" in c.source:
            continue                                    # packages come from the test env; lasp_lab.py is local
        if c.cell_type == "code" and DEPLOYED_URL in c.source and os.environ.get("LASP_MCP_URL"):
            c.source = c.source.replace(DEPLOYED_URL, os.environ["LASP_MCP_URL"])
        if c.cell_type == "code" and 'PROVIDER = "' in c.source:
            c.source = re.sub(r'PROVIDER = "\w+"', f'PROVIDER = "{PROVIDER}"', c.source)
        cells.append(c)
        if not LIVE and c.cell_type == "code" and "init_chat_model(MODEL" in c.source:
            cells.append(nbformat.v4.new_code_cell(FAKE_MODEL))
        if c.cell_type == "code" and "from lasp_lab import *" in c.source:
            cells.append(nbformat.v4.new_code_cell(FAKE_DRAW))
    nb.cells = cells
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": HERE}}).execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    out = os.path.join(os.environ.get("NB_OUT", "/tmp"), "executed-" + name)
    nbformat.write(nb, out)
    return out, errors


if __name__ == "__main__":
    os.environ.setdefault("LASP_MCP_TOKEN", "dev-token")
    if not LIVE:
        os.environ.setdefault("GOOGLE_API_KEY", "not-used")
        os.environ.setdefault("ANTHROPIC_API_KEY", "not-used")
    names = [a for a in sys.argv[1:] if a.endswith(".ipynb")] or sorted(
        f for f in os.listdir(HERE) if f.endswith(".ipynb"))
    failed = 0
    for name in names:
        try:
            out, errors = run(name)
            print(f"{'✅' if not errors else '❌'} {name}  ->  {out}")
        except Exception as e:
            failed += 1
            print(f"❌ {name}: {type(e).__name__}: …{str(e)[-1200:]}")
    sys.exit(1 if failed else 0)
