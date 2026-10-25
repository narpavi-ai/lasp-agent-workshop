<!-- The adsorbent_screening playbook, copied from the private lasp-mcp-server repo
     (src/server.js). n8n's MCP nodes can't read MCP prompts, so build_workflows.py
     pastes this into the agent's system message. Keep it in step with the server. -->
Screen this adsorbent for ${gas1}/${gas2} separation at y_${gas1} = ${y1}, P = ${pressure_bar} bar.

CALL PLAN
1. fit_isotherm once per gas (for a bundled dataset, get_sample_dataset first). Never refit data you
   already fitted: the results are in your context.
2. iast_binary once with your chosen pair, passing each fit's iast_handoff entry unchanged.
3. Only if a choice below is tied: one more iast_binary per other tied model (at most 3 IAST calls in all),
   to report the range.

CHOOSING A FIT (the same rule for any data)
- Use only fits that have an iast_handoff and whose workshop_validation status is not "fail".
- Fits within 2 AICc units of the best usable one are tied. Among tied fits, choose the one with the fewest
  parameters (n_params); if still tied, the higher-ranked one.

ANSWER IN EXACTLY THIS FORMAT
| gas | model | n_params | AICc gap to the next usable fit | why |
(one row per gas)
**Selectivity S = <value>** at y_${gas1} = ${y1}, P = ${pressure_bar} bar (range <min>-<max> across tied fits, or "no tie")
Henry selectivity: <value>
Warnings: every warn/fail check, with its numbers, or "none"
Caveat: one sentence; if IAST extrapolates beyond the fitted range, say the answer depends on the isotherm there.
