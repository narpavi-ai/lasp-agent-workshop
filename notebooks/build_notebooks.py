#!/usr/bin/env python3
"""Generate the Colab lab notebooks.

Why a generator: setup cells, pinned versions and the server URL are shared by
every lab. Write them once here, change workshop.json, re-run. The helper code
the notebooks import lives in lasp_lab.py, which each notebook downloads.

    python3 notebooks/build_notebooks.py
"""
import json
import os
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "..", "workshop.json")))
MODELS = {"gemini": f"google_genai:{CFG['models']['gemini']}", "claude": f"anthropic:{CFG['models']['claude']}"}
PROVIDER = CFG["default_provider"]
RPS = round(CFG["gemini_requests_per_minute"] / 60, 3)
RAW = f"https://raw.githubusercontent.com/{CFG['github_repo']}/main"
MCP_URL = CFG["mcp_url"]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(s).strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(s).strip("\n")}


def badge(name):
    url = f"https://colab.research.google.com/github/{CFG['github_repo']}/blob/main/notebooks/{name}"
    return md(f'<a href="{url}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>')


def title(n, name, intro):
    return md(f"# Lab {n} · {name}\n\n**LASP workshop: build your agent**\n\n" + textwrap.dedent(intro).strip("\n"))


# ---------------------------------------------------------------- shared data

P = [0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 5, 7, 10]
CO2 = [0.147171, 0.399272, 1.02996, 1.9223, 2.88179, 3.25169, 3.86714, 4.16431, 4.24648,
       4.41152, 4.66168, 4.89841, 5.10685, 5.2541, 5.17341, 5.23311]
N2 = [0.000319116, 0.000968288, 0.00320824, 0.00947269, 0.0311209, 0.0631523, 0.122319,
      0.186175, 0.241094, 0.29566, 0.436553, 0.552997, 0.78373, 1.14108, 1.43167, 1.77024]

DATA = f'''# Synthetic 13X-like data made for this workshop (not measured), 298 K.
# The same numbers are on the server as workshop_co2_298K_synthetic and workshop_n2_298K_synthetic.
pressure_bar = {P}
co2_mol_per_kg = {CO2}
n2_mol_per_kg = {N2}
'''

QUESTION = ('USER_PROMPT = (f"Here are pure-component CO2 and N2 isotherms at 298 K on a 13X-like adsorbent.\\n"\n'
            '               f"pressure (bar): {pressure_bar}\\nCO2 loading (mol/kg): {co2_mol_per_kg}\\n"\n'
            '               f"N2 loading (mol/kg): {n2_mol_per_kg}\\n\\n"\n'
            '               "Using IAST, what is the CO2/N2 selectivity for a flue gas with 15% CO2 at 1 bar total pressure?")')

LASP_PROMPT = ('"You are a research assistant at the Laboratory for Advanced Separation Processes (LASP), "\n'
               '                 "University of Alberta, with the LASP Isotherm Fitter and IAST tools. "\n'
               '                 "Never estimate a number a tool can compute. Report the models you used and any "\n'
               '                 "workshop_validation warnings."')

SCREEN = ("Screen the bundled workshop_co2_298K_synthetic and workshop_n2_298K_synthetic datasets "
          "for CO2/N2 separation at 15% CO2 and 1 bar.")

# ---------------------------------------------------------------- setup cells

# Exact versions, as tested. A release the night before can't break the class.
PINS_MODEL = "langchain==1.4.7 langchain-core==1.7.0 langchain-google-genai==4.4.2 langchain-anthropic==1.7.6"
PINS_AGENT = PINS_MODEL + " langgraph==1.2.17 langchain-mcp-adapters==0.3.2 mcp==1.30.0"


def install(pins):
    # test_notebooks.py skips this cell: it has its own environment and the local lasp_lab.py.
    return code(f'''
# pip may print a warning about google-auth and google-colab. That's expected
# and harmless: Gemini's library needs a newer google-auth than Colab ships.
%pip install -q {pins}
!curl -sSfO {RAW}/notebooks/lasp_lab.py
print("✅ Installed, and helpers downloaded.")
''')


INSTALL_MODEL = install(PINS_MODEL)
INSTALL_AGENT = install(PINS_AGENT)

HELPERS = code('''
from lasp_lab import *   # show(), call(), run_live() and friends: see lasp_lab.py
print("✅ Helpers ready.")
''')

PROVIDER_CELL = code(f'''
PROVIDER = "{PROVIDER}"    # "gemini" or "claude"
''')

KEYS = code('''
import os
KEY_NAME = {"gemini": "GOOGLE_API_KEY", "claude": "ANTHROPIC_API_KEY"}[PROVIDER]
os.environ[KEY_NAME] = secret(KEY_NAME)
print(f"✅ {KEY_NAME} loaded.")
''')

MODEL_CELL = code(f'''
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter

MODELS = {{"gemini": "{MODELS['gemini']}", "claude": "{MODELS['claude']}"}}
MODEL = MODELS[PROVIDER]

# Gemini's free tier allows 15 requests a minute; stay under it. Paid Claude keys allow more.
limiter = InMemoryRateLimiter(requests_per_second={RPS}) if PROVIDER == "gemini" else None

# Claude 5 thinks before answering. Low effort keeps that short (faster, cheaper),
# and the cap leaves room for the answer after the thinking.
extra = {{"max_tokens": 16000, "thinking": {{"type": "adaptive"}}, "output_config": {{"effort": "low"}}}} if PROVIDER == "claude" else {{}}
model = init_chat_model(MODEL, rate_limiter=limiter, **extra)
print(f"✅ Model ready: {{MODEL}}")
''')

CONNECT = code(f'''
LASP_MCP_URL = "{MCP_URL}"   # where the LASP server lives
LASP_MCP_TOKEN = secret("LASP_MCP_TOKEN")   # the workshop token, from Colab Secrets

client, tools = await connect(LASP_MCP_URL, LASP_MCP_TOKEN)
print(f"✅ Connected to the LASP server: {{len(tools)}} tools")
''')

AGENTS = code(f'''
from langchain.agents import create_agent

# The server's screening playbook: Lab 3's skill
skill = (await client.get_prompt("lasp", "adsorbent_screening",
                                 arguments={{"gas1": "CO2", "gas2": "N2", "y1": "0.15", "pressure_bar": "1"}}))[0].content

agent_tools = [t for t in tools if not t.name.startswith("plot_")]
SYSTEM_PROMPT = ({LASP_PROMPT})

plain = create_agent(model, agent_tools, system_prompt=SYSTEM_PROMPT)
skilled = create_agent(model, agent_tools, system_prompt=SYSTEM_PROMPT + "\\n\\nFollow this procedure:\\n" + skill)
print("✅ Two agents: plain, and skilled (with the playbook).")
''')

SETUP_TEXT = {}


def setup(*cells):
    """Setup cells, each under a short '### Setup · …' heading with one line of description."""
    out = [md("## Setup\n\nRun these in order. Each takes a few seconds.")]
    for c in cells:
        t, d = SETUP_TEXT[id(c)]
        out += [md(f"### Setup · {t}\n\n{d}"), c]
    return out


for cell, t, d in [
    (INSTALL_MODEL, "Install", "LangChain with its Gemini and Claude connectors, at the exact versions we tested, "
                               "plus this workshop's helper file."),
    (INSTALL_AGENT, "Install", "LangChain talks to the model, LangGraph runs agents, and the MCP adapter connects "
                               "to the LASP server. Exact versions, as tested, plus this workshop's helper file."),
    (HELPERS, "Helpers", "Display and connection helpers, so the notebook shows only what's being taught. "
                         "Curious? Read [`lasp_lab.py`](lasp_lab.py)."),
    (PROVIDER_CELL, "Choose your model", '`"gemini"` is free and needs `GOOGLE_API_KEY`. `"claude"` is paid and needs '
                                         '`ANTHROPIC_API_KEY`. Nothing else in the lab changes.'),
    (KEYS, "Load your key", "Read from Colab Secrets, so it never appears in the notebook."),
    (MODEL_CELL, "Create the model", "One line picks the provider and model. Everything after it is the same for both."),
    (CONNECT, "Connect to the LASP server", "The same connection as Lab 2: the server's address, and the workshop token."),
    (AGENTS, "The agents", "Lab 3's two agents: the same model and tools, with and without the screening skill."),
]:
    SETUP_TEXT[id(cell)] = (t, d)


def keys_md(server):
    rows = ["| `GOOGLE_API_KEY` | your free key from <https://aistudio.google.com/apikey> |"]
    if server:
        rows.append("| `LASP_MCP_TOKEN` | the workshop token, shown on the slide |")
    rows.append("| `ANTHROPIC_API_KEY` | *optional:* only if you switch to Claude (a paid key from <https://console.anthropic.com>) |")
    return md("## Before you start\n\nAdd these with the **🔑 key icon** in Colab's left sidebar, and turn **Notebook "
              "access** on for each. Notebook access is per notebook, so each lab asks again.\n\n"
              "| Name | Value |\n|---|---|\n" + "\n".join(rows))


def breaks(server=True):
    rows = [
        "| `404 NOT_FOUND`, model not available | The provider retired the model. Pick a current one "
        "(<https://aistudio.google.com> or <https://docs.anthropic.com/en/docs/about-claude/models>) and change it in `MODELS`. |",
        "| `401`, `API key not valid`, `authentication_error` | Check the key's secret (`GOOGLE_API_KEY` or "
        "`ANTHROPIC_API_KEY`): exact name, Notebook access on, and that `PROVIDER` matches it. |",
    ]
    if server:
        rows.append("| `Couldn't connect to …` | The message says why. A `401` means the `LASP_MCP_TOKEN` secret is wrong: "
                    "48 characters, `0-9` and `a-f` (a trailing `%` copied from the terminal isn't part of it). |")
    rows += ["| `429`, `RESOURCE_EXHAUSTED` | Free-tier limit. Wait a minute and re-run the cell. |",
             "| `ModuleNotFoundError`, or a name not defined | The runtime restarted. Run the setup cells again from the top. |"]
    return md("## If something breaks\n\n| What you see | Fix |\n|---|---|\n" + "\n".join(rows))


# ====================================================================== Lab 1

def lab1():
    name = "01-plain-model.ipynb"
    return name, [
        badge(name),
        title(1, "Ask the model", """
        You have pure-component CO₂ and N₂ isotherms for a 13X-like adsorbent.
        You want one number: **the CO₂/N₂ selectivity for flue gas (15% CO₂) at
        1 bar.** In the lab you'd fit both isotherms, then run IAST.

        Here, you ask a language model, and find out exactly what it can and
        can't do. About **5 requests**.
        """),
        keys_md(server=False),
        *setup(INSTALL_MODEL, HELPERS, PROVIDER_CELL, KEYS, MODEL_CELL),
        md("""
        ### Step 1 · One call: a system prompt and a user prompt

        Every call sends a list of **messages**, each with a role:

        | Role | Who writes it | What it's for |
        |---|---|---|
        | `system` | you, the developer | standing instructions: who the model is, what rules it follows |
        | `user` | the person asking | the question |

        What comes back is a response object, not just text.
        """),
        code('''
        SYSTEM_PROMPT = ("You are a research assistant at the Laboratory for Advanced Separation Processes (LASP), "
                         "led by Prof. Arvind Rajendran at the University of Alberta. The group works on "
                         "adsorption-based separations such as CO2 capture.")
        USER_PROMPT = "In two sentences: what does Ideal Adsorbed Solution Theory assume?"

        response = model.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ])
        show(response, "Model answer", "A well-documented concept, so this is where a language model is strong.")
        print(response.usage_metadata)   # what this call cost, in tokens
        '''),
        md("""
        ### Step 2 · The real question

        Same model and system prompt. Now the user prompt carries **your data**,
        and the answer depends on it.
        """),
        code(DATA + "\n" + QUESTION + '''

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT},
]
first = model.invoke(messages)
show(first, "Attempt 1", "Read it critically: is there a number? How did it get it?")
'''),
        md("""
        Note what it gave you: a number, a range, or only a method.

        Now send **exactly the same messages** again. The model doesn't remember
        attempt 1: every call starts from nothing, because **the model is
        stateless**. Chat apps only seem to remember because they resend the
        whole conversation each time.
        """),
        code('''
        second = model.invoke(messages)
        show(second, "Attempt 2", "Same messages, fresh call, no memory of attempt 1. Does it agree?")
        '''),
        md("""
        ### What just happened

        Running LASP's own Isotherm Fitter and IAST engines on this data gives
        **S ≈ 388–420**: 388 if you fit N₂ with Langmuir, and up to 420 with the
        other two N₂ models that fit equally well. You'll compute this in Lab 2.

        Your answers will differ from run to run. Here's what the models said to
        this exact question **in our test runs while preparing this workshop**
        (October 2026):

        | Model | Where | Answer |
        |---|---|---|
        | Gemini 3.5 Flash Lite | n8n | **25.6**, stated with confidence |
        | Claude Haiku 5.5 | n8n | **≈ 115**, "a hand estimate" |
        | Claude Haiku 5.5 | Colab, attempt 1 | **≈ 480**, "a hand-calculation estimate" |
        | Claude Haiku 5.5 | Colab, attempt 2 | **≈ 390**, and it flagged the N₂ extrapolation |

        Three things to take from that:

        - **A wrong answer and a good one look equally confident.** 25.6 is off by
          a factor of 15, and nothing in its tone says so.
        - **Even a careful answer isn't reproducible.** The same model gave 115,
          480 and 390.
        - **Even when it's close, you can't check it.** Which fit did it use? How
          did it integrate the spreading pressure? You'd have to redo the work to
          trust the number.

        ## The gap, and what closes it

        | A model can | A model can't |
        |---|---|
        | explain IAST, outline the method, reason about which model to use | run *your* solver, or give the same checked number twice |

        **A tool closes the gap.** In Lab 2 the same model gets the LASP apps as
        tools, through an MCP server, so the number comes from the solver.

        ### Your turn

        Change only the system prompt. Can you make the model honest about what it
        can't compute?
        """),
        code('''
        STRICT_SYSTEM_PROMPT = "Never give a number you didn't compute with a tool. Say what you'd need instead."

        show(model.invoke([
            {"role": "system", "content": STRICT_SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]), "With a stricter system prompt", "Same user prompt; only the system prompt changed.")
        '''),
        md("""
        A system prompt can make the model **honest about the gap**. It can't
        **close** it.
        """),
        breaks(server=False),
    ]


# ====================================================================== Lab 2

def lab2():
    name = "02-mcp-and-agent.ipynb"
    return name, [
        badge(name),
        title(2, "MCP, and an agent", """
        Same question as Lab 1. This time the model can use the LASP Isotherm
        Fitter and IAST apps as **tools**, served by an **MCP server**. The tools
        run the apps' own engines, unchanged.

        1. **What MCP is**, and connecting to the LASP server.
        2. **You are the workflow:** you call the tools yourself, step by step.
        3. **An agent:** the model decides the steps.

        About **12 requests**.
        """),
        keys_md(server=True),
        *setup(INSTALL_AGENT, HELPERS, PROVIDER_CELL, KEYS, MODEL_CELL),
        md("""
        ### Step 1 · What is MCP?

        The **Model Context Protocol** is a standard way to give AI models tools.
        A **server** offers tools; any **client** can use them: this notebook,
        n8n, Claude Desktop, an IDE. Write the tools once, use them everywhere.
        It's often described as "USB-C for AI tools".

        The LASP server runs on Google Cloud. To connect you need two things:

        | Placeholder | What it is | Where it comes from |
        |---|---|---|
        | `LASP_MCP_URL` | the server's address | already filled in below |
        | `LASP_MCP_TOKEN` | the workshop password | your Colab secret (the token on the slide) |

        An MCP server **describes its own tools**. Each description is what the
        model reads when it decides which tool to use.
        """),
        code(f'''
        LASP_MCP_URL = "{MCP_URL}"   # where the LASP server lives
        LASP_MCP_TOKEN = secret("LASP_MCP_TOKEN")   # the workshop token, from Colab Secrets

        client, tools = await connect(LASP_MCP_URL, LASP_MCP_TOKEN)

        for t in tools:
            print(f"{{t.name:22s}} {{t.description[:90]}}…")
        '''),
        md("""
        Eight tools: the Isotherm Fitter (`fit_isotherm`, `plot_isotherm_fit`),
        IAST (`iast_binary`, `iast_sweep_summary`, `iast_xy_diagram`, `plot_iast`),
        and two helpers. Results also carry `workshop_validation`, physical checks
        that are a **workshop addition**, not part of the LASP apps.

        ### Step 2 · You are the workflow

        A tool is just a function on a server; no model needed. Do what you'd do
        in the lab: fit both isotherms with all 16 of the Isotherm Fitter's models
        (ranked by AICc, as the app ranks them).
        """),
        code(DATA + '''
co2_fit = await call("fit_isotherm", pressure=pressure_bar, loading=co2_mol_per_kg)
n2_fit = await call("fit_isotherm", pressure=pressure_bar, loading=n2_mol_per_kg)
show_fit(co2_fit, "CO₂ fit")
show_fit(n2_fit, "N₂ fit")
await call("plot_isotherm_fit", pressure=pressure_bar, loading=n2_mol_per_kg, panel="fits");
'''),
        md("""
        That plot is the app's "Data & fits" panel for N₂: the top five models sit
        almost on top of each other. Now take each gas's top-ranked fit and run IAST
        for flue gas.
        """),
        code('''
        def top_iast_fit(fit):
            """The rule we chose: the highest-ranked fit that IAST can use."""
            return next(f for f in fit["ranked"] if f.get("iast_handoff"))

        co2_model, n2_model = top_iast_fit(co2_fit), top_iast_fit(n2_fit)
        print("Chosen:", co2_model["model"], "for CO2,", n2_model["model"], "for N2")

        iast = await call("iast_binary", component1=co2_model["iast_handoff"][0], component2=n2_model["iast_handoff"][0],
                          total_pressure_bar=[1], y1=[0.15])
        show_iast(iast)
        await call("plot_iast", component1=co2_model["iast_handoff"][0], component2=n2_model["iast_handoff"][0],
                   panel="loadings_vs_P", y1=0.15);
        '''),
        md("""
        **A real number, from the real solver,** and no model involved. The yellow
        band marks where IAST needs N₂'s isotherm beyond the 10 bar it was fitted
        to, which is the extrapolation warning.

        This is a **workflow**: fixed steps that you decided in advance (the n8n
        track builds the same thing as a chain of nodes). It's predictable, and it
        does exactly what you wrote. Notice what the rule did: it picked **Toth**
        for N₂ (S ≈ 414). But Toth, Quadratic and Langmuir are within 1.5 AICc
        units, a statistical tie, and give S = 414, 420 and 388. The workflow
        can't notice that, because you didn't write a step for it.

        ### Step 3 · An agent

        An agent replaces your fixed steps with a **loop**: the model either calls
        a tool or answers. `create_agent` builds it from a model, the tools, and a
        **system prompt**, the agent's standing instructions.
        """),
        code(f'''
        from langchain.agents import create_agent

        agent_tools = [t for t in tools if not t.name.startswith("plot_")]   # images cost tokens
        SYSTEM_PROMPT = ({LASP_PROMPT})

        agent = create_agent(model, agent_tools, system_prompt=SYSTEM_PROMPT)
        draw(agent)
        '''),
        md("""
        **That's the loop:** `model → tools → model → …` until the model answers
        instead of calling a tool. LangGraph runs it.

        Now the **user prompt**: Lab 1's question, with the data in it.
        """),
        code(QUESTION + '''

messages = await run_live(agent, USER_PROMPT)
'''),
        md("""
        Read the trace. The model chose every step: which tool, which arguments,
        and when to stop. Did it copy your data correctly into `fit_isotherm`?
        Which N₂ model did it pick, and did it say why? If a workshop check
        flagged a mistake, did it notice and retry?

        ### Step 4 · A question your workflow can't answer
        """),
        code('''
        FOLLOW_UP = (USER_PROMPT + "\\n\\nSeveral N2 models may fit about equally well. "
                     "Does the choice of N2 model change the selectivity? Compare the top three IAST-compatible N2 fits.")
        messages = await run_live(agent, FOLLOW_UP)
        '''),
        md("""
        Your Step 2 code would need new steps for this. The agent planned it from
        the question. That flexibility is the point of an agent, and also its
        risk: **you didn't decide the steps, so you have to check them.** That's
        Labs 3 and 4.

        ### Step 5 · The safety belt

        An agent loops until it decides to stop. `recursion_limit` caps the
        steps, so a confused agent can't loop and spend forever.
        """),
        code('''
        from langgraph.errors import GraphRecursionError
        try:
            await run_live(agent, FOLLOW_UP, recursion_limit=4)
        except GraphRecursionError:
            print("🛑 Stopped by recursion_limit, as intended.")
        '''),
        md("""
        ## What you built

        | | Workflow (Step 2) | Agent (Steps 3–4) |
        |---|---|---|
        | Who decides the steps | you, in code | the model, at run time |
        | Model requests | none | one per step |
        | Predictable | yes | no, so you evaluate it |
        | Handles a new question | only if you wrote steps for it | yes |

        **In the n8n track** the workflow is MCP Client nodes in a row, and the
        agent is the AI Agent node with an MCP Client Tool.
        """),
        breaks(),
    ]


# ====================================================================== Lab 3

def lab3():
    name = "03-skill.ipynb"
    return name, [
        badge(name),
        title(3, "A skill", """
        Lab 2's agent got real numbers, but nobody told it how to do the job
        *well*. What should it do when fits are tied? Which warnings must it
        always report?

        A **skill** is a written procedure the agent follows. Here it ships with
        the tools, on the MCP server. About **10 requests**.
        """),
        keys_md(server=True),
        *setup(INSTALL_AGENT, HELPERS, PROVIDER_CELL, KEYS, MODEL_CELL, CONNECT),
        md("""
        ### Step 1 · The skill

        Besides tools, an MCP server can offer **prompts**: reusable instructions
        written by whoever knows the domain. Read the one the LASP server ships:
        """),
        code('''
        skill = (await client.get_prompt("lasp", "adsorbent_screening",
                                         arguments={"gas1": "CO2", "gas2": "N2", "y1": "0.15", "pressure_bar": "1"}))[0].content
        show_text(skill, "adsorbent_screening", icon="📋")
        '''),
        md("""
        It has three parts, and each fixes something you saw in Lab 2:

        - **A call plan:** which tools, how many times. No re-fitting the same data.
        - **A rule for choosing:** the same rule for any data, not an answer for
          this data. It never says "use Toth"; it says how to break a tie.
        - **An output format:** every answer has the same shape, so you (and
          Lab 4's judge) can compare runs.

        That's a **skill**: a procedure, not code. The same idea appears as Agent
        Skills (a folder with a `SKILL.md`) in Claude and other agent tools. Here
        it travels with the server, so whoever maintains the tools maintains the
        know-how too.

        ### Step 2 · Two agents

        Same model, same tools. The only difference is the system prompt: one has
        the skill appended.
        """),
        code(f'''
        from langchain.agents import create_agent

        agent_tools = [t for t in tools if not t.name.startswith("plot_")]
        SYSTEM_PROMPT = ({LASP_PROMPT})

        plain = create_agent(model, agent_tools, system_prompt=SYSTEM_PROMPT)
        skilled = create_agent(model, agent_tools, system_prompt=SYSTEM_PROMPT + "\\n\\nFollow this procedure:\\n" + skill)
        print("✅ Two agents: plain, and skilled.")
        '''),
        md("""
        ### Step 3 · Same question, both agents

        This time the agent fetches the data itself from the server. It's **the
        same data as Lab 2**: the bundled datasets hold exactly the numbers Lab 2
        pasted into the prompt.
        """),
        code(f'''
        USER_PROMPT = "{SCREEN}"

        plain_messages = await run_live(plain, USER_PROMPT)
        '''),
        code('''
        skilled_messages = await run_live(skilled, USER_PROMPT)
        '''),
        md("""
        ### Compare

        - **Tool calls:** count them in each trace. Did the skilled agent follow
          the call plan?
        - **The choice:** Toth, Quadratic and Langmuir tie for N₂. The rule picks
          Langmuir (fewest parameters) and reports the range, S ≈ 388–420. What
          did the plain agent pick, and did it say why?
        - **The format:** is the skilled answer in the table-and-lines format?
          Run it again: is the shape the same?
        - Did it **report the extrapolation warning** with numbers?

        "It looks better" isn't evidence, though. One run is one sample, and you
        compared by eye. **Lab 4 measures it.**
        """),
        breaks(),
    ]


# ====================================================================== Lab 4

def lab4():
    name = "04-evaluation.ipynb"
    return name, [
        badge(name),
        title(4, "Evaluation", """
        Lab 3's skilled agent *looked* better. This lab measures it with an
        **LLM-as-judge**: a second model reads each answer and scores it against
        written criteria, using **reference facts computed by LASP's solver**.

        Both agents answer the same question once; the judge scores each answer on
        three criteria. About **16 requests**.
        """),
        keys_md(server=True),
        *setup(INSTALL_AGENT, HELPERS, PROVIDER_CELL, KEYS, MODEL_CELL, CONNECT, AGENTS),
        md("""
        ### Step 1 · Reference facts, from the solver

        A judge needs to know the right answer, or it grades on style. Here the
        facts come from **LASP's engines**: call the tools directly, as in Lab 2.
        """),
        code('''
        co2_data = await call("get_sample_dataset", name="workshop_co2_298K_synthetic")
        n2_data = await call("get_sample_dataset", name="workshop_n2_298K_synthetic")
        co2_fit = await call("fit_isotherm", pressure=co2_data["pressure_bar"], loading=co2_data["loading_mol_per_kg"])
        n2_fit = await call("fit_isotherm", pressure=n2_data["pressure_bar"], loading=n2_data["loading_mol_per_kg"])

        def handoff(fit, model_id):
            return next(f for f in fit["ranked"] if f["model"] == model_id)["iast_handoff"][0]

        co2_best = co2_fit["ranked"][0]["model"]
        tied_n2 = [f["model"] for f in n2_fit["ranked"] if f.get("iast_handoff")][:3]   # the top three N2 fits

        S = {}
        for m in tied_n2:
            point = (await call("iast_binary", component1=handoff(co2_fit, co2_best), component2=handoff(n2_fit, m),
                                total_pressure_bar=[1], y1=[0.15]))["points"][0]
            S[m], n2_pressure = point["selectivity_S12"], point["hypothetical_pressures_bar"][1]

        REFERENCE = (f"Facts computed by LASP's solver for this question:\\n"
                     f"- CO2 is best fitted by {co2_best}. For N2, {', '.join(tied_n2)} fit about equally well.\\n"
                     f"- IAST selectivity at 1 bar and 15% CO2, by N2 model: "
                     + ", ".join(f"{m} {s:.0f}" for m, s in S.items()) + ".\\n"
                     f"- IAST needs the N2 isotherm at about {n2_pressure:.0f} bar; N2 was fitted only up to 10 bar.")
        show_text(REFERENCE, "Reference facts", icon="📐")
        '''),
        md("""
        ### Step 2 · The judge

        The judge is just another model call: the reference facts, one criterion,
        the answer to grade, and a request for a score from 0 to 1 with a reason.
        """),
        code(f'''
        import json, re

        JUDGE = model   # or a different model, e.g. init_chat_model("{MODELS['claude']}") to have Claude judge Gemini

        CRITERIA = {{
            "selectivity": "Gives a selectivity within the reference values and says which isotherm models it used.",
            "extrapolation": "Says IAST uses the N2 isotherm far beyond its fitted range, so the result is uncertain.",
            "tied models": "Notes that several N2 models fit about equally well, and compares their selectivities "
                           "or justifies the one it chose.",
        }}

        def judge(answer, criterion):
            prompt = (f"You are grading an AI research assistant's answer.\\n\\n{{REFERENCE}}\\n\\n"
                      f"Criterion: {{criterion}}\\n\\nAnswer to grade:\\n{{answer}}\\n\\n"
                      "Score from 0.0 (criterion not met) to 1.0 (fully met). "
                      'Reply with JSON only: {{"score": <number>, "reason": "<one sentence>"}}')
            reply = text_of(JUDGE.invoke(prompt))
            try:
                verdict = json.loads(re.search(r"\\{{.*\\}}", reply, re.S).group(0))   # tolerate text around the JSON
                return {{"score": min(1.0, max(0.0, float(verdict["score"]))), "reason": verdict.get("reason", "")}}
            except (AttributeError, KeyError, ValueError, TypeError):
                return {{"score": 0.0, "reason": f"unreadable verdict: {{reply[:80]}}"}}
        '''),
        md("""
        ### Step 3 · Both agents answer, the judge scores

        Lab 3's question, once per agent. Each answer is shown, so you can check
        the judge's scores against your own reading.
        """),
        code(f'''
        USER_PROMPT = "{SCREEN}"

        answers, rows = {{}}, []
        for name, agent in [("plain", plain), ("skilled", skilled)]:
            answers[name] = answer_of(await run_live(agent, USER_PROMPT, quiet=True))
            show(answers[name], f"{{name}} agent", "The answer the judge is about to score.")
            for criterion, text in CRITERIA.items():
                rows.append({{"agent": name, "criterion": criterion, **judge(answers[name], text)}})
        '''),
        md("### Step 4 · The scorecard"),
        code('''
        import pandas as pd
        from IPython.display import display

        results = pd.DataFrame(rows)
        display(results.pivot(index="criterion", columns="agent", values="score").style.format("{:.1f}"))
        display(results[["agent", "criterion", "score", "reason"]])
        '''),
        md("""
        ### Reading the scorecard

        - **Did the skill help?** Compare the two columns, especially
          *extrapolation* and *tied models*: that's what the playbook asks for.
        - **Check the judge.** Read its reasons against the answers above. Do you
          agree with every score?
        - **One run is one sample.** Run it again and some scores will change. A
          trustworthy evaluation uses more questions and several runs.

        ### The judge's limits

        A judge is also a model: it can be wrong, it costs a request per score,
        and models tend to grade their own style kindly. That's why this judge
        gets the solver's facts, specific criteria, and gives a reason for every
        score. When there's a single right answer (a model id, a number in a
        range), plain code is cheaper and exact; a judge earns its keep on things
        code can't check, like whether a caveat was reported.

        ### Your turn

        - Switch `JUDGE` to the other provider. Do the scores change?
        - Add a criterion to `CRITERIA`, for example *"Gives the Henry's-law
          selectivity."* Re-run Steps 3 and 4.

        **Going further:** [`openevals`](https://github.com/langchain-ai/openevals)
        has ready-made LLM-as-judge prompts, and [LangSmith](https://docs.smith.langchain.com/)
        stores runs and scores over time. **In the n8n track**, the same judge is a
        Basic LLM Chain scoring the agent's answer.
        """),
        breaks(),
    ]


def main():
    for old in ("02-chain-and-agent.ipynb", "03-skill-and-evaluation.ipynb", "04-bonus-your-own-tool.ipynb"):
        if os.path.exists(os.path.join(HERE, old)):
            os.remove(os.path.join(HERE, old))
    for build in (lab1, lab2, lab3, lab4):
        name, cells = build()
        nb = {"cells": cells, "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
        with open(os.path.join(HERE, name), "w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"wrote notebooks/{name}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
