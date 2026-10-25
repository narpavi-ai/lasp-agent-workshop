#!/usr/bin/env python3
"""Structural check on the generated n8n workflows (adapted from CCTP 481).

This doesn't prove a workflow imports; only a live n8n can do that. It catches
the mistakes that are easy to make and invisible until class: a connection to
a node that doesn't exist, a sub-node on the wrong connection type, an agent
with no model, an MCP node with no URL, tool or credential.

    python3 n8n/validate_workflows.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = "@n8n/n8n-nodes-langchain"

SUBNODE_CONNECTION = {
    f"{PKG}.lmChatGoogleGemini": "ai_languageModel",
    f"{PKG}.lmChatAnthropic": "ai_languageModel",
    f"{PKG}.memoryBufferWindow": "ai_memory",
    f"{PKG}.mcpClientTool": "ai_tool",
}
ROOT_NODES = {f"{PKG}.agent", f"{PKG}.chainLlm"}
MCP_NODES = {f"{PKG}.mcpClient", f"{PKG}.mcpClientTool"}


def check(path):
    wf = json.load(open(path))
    errs, warns = [], []
    nodes = wf.get("nodes", [])
    by_name = {}
    for n in nodes:
        for key in ("name", "type", "typeVersion", "position", "parameters"):
            if key not in n:
                errs.append(f"node {n.get('name', '?')!r} missing {key!r}")
        if n["name"] in by_name:
            errs.append(f"duplicate node name {n['name']!r}")
        by_name[n["name"]] = n

    conns = wf.get("connections", {})
    for src, kinds in conns.items():
        if src not in by_name:
            errs.append(f"connection from unknown node {src!r}")
            continue
        for kind, branches in kinds.items():
            for branch in branches:
                for link in branch:
                    if link["node"] not in by_name:
                        errs.append(f"{src!r} -> unknown node {link['node']!r}")
                    if link.get("type") != kind:
                        errs.append(f"{src!r}: connection kind {kind!r} but link type {link.get('type')!r}")
            expected = SUBNODE_CONNECTION.get(by_name[src]["type"])
            if expected and kind != expected:
                errs.append(f"{src!r} wired as {kind!r}, expected {expected!r}")

    for n in nodes:
        if n["type"] in ROOT_NODES:
            fed = [s for s, k in conns.items() if "ai_languageModel" in k
                   and any(l["node"] == n["name"] for b in k["ai_languageModel"] for l in b)]
            if not fed:
                errs.append(f"{n['name']!r} has no ai_languageModel connected")
        if n["type"] in MCP_NODES:
            p = n["parameters"]
            if not p.get("endpointUrl", "").startswith("http"):
                errs.append(f"MCP node {n['name']!r} has no endpoint URL")
            elif "REPLACE-AFTER-DEPLOY" in p["endpointUrl"]:
                warns.append(f"MCP node {n['name']!r} still has the placeholder URL (set mcp_url in workshop.json)")
            if p.get("authentication") == "bearerAuth" and "httpBearerAuth" not in n.get("credentials", {}):
                errs.append(f"MCP node {n['name']!r} uses bearer auth but has no httpBearerAuth credential slot")
            if n["type"] == f"{PKG}.mcpClient" and not p.get("tool", {}).get("value"):
                errs.append(f"MCP node {n['name']!r} names no tool")
        # a node referenced by $('Name') in an expression or code must exist
        blob = json.dumps(n["parameters"])
        for ref in set(__import__("re").findall(r"\$\('([^']+)'\)", blob)):
            if ref not in by_name:
                errs.append(f"{n['name']!r} references unknown node $('{ref}')")

    if not any(n["type"].endswith(".stickyNote") for n in nodes):
        warns.append("no sticky note: attendees get no on-canvas explanation")
    return errs, warns


def main():
    files = sorted(glob.glob(os.path.join(HERE, "*", "*.json")))
    if not files:
        sys.exit("no workflow JSON found; run build_workflows.py first")
    total = 0
    for path in files:
        errs, warns = check(path)
        total += len(errs)
        print(f"{'FAIL' if errs else 'ok  '} {os.path.relpath(path, HERE)}")
        for e in errs:
            print(f"       ERROR {e}")
        for w in warns:
            print(f"       warn  {w}")
    print(f"\n{len(files)} workflow(s), {total} error(s). Structure only: import into a real n8n before class.")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
