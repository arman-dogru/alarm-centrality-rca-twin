"""Render the README's result tables from results/summary.json (no hand-copied numbers).

Usage: python scripts/readme_tables.py [--write]   (--write replaces the <!-- BEGIN:x --> ... <!-- END:x --> blocks)
"""
import json, re, sys

S = json.load(open("results/summary.json"))
M = S["phases"]["main"]
KINDS = ["hierarchical", "star", "mesh", "hybrid"]
NAMES = {"random_candidate": "random candidate in cascade", "global_pagerank": "global PageRank (no cascade)",
         "earliest_alarm": "earliest alarm (raw)", "earliest_in_cascade": "earliest alarm in cascade",
         "common_ancestor": "common upstream ancestor", "static_pagerank": "static PageRank on cascade",
         "static_degree": "static degree on cascade", "static_closeness": "static closeness on cascade",
         "ppr": "**alarm-seeded PPR**", "alarm_degree": "**alarm-degree**", "alarm_closeness": "**alarm-closeness**",
         "alarm_degree_tb": "alarm-degree + closeness tie-break (exploratory)", "equal": "**trio, equal weights**",
         "triage_blog_select": "triage -> blog mapping, select one", "triage_blog_weighted": "triage -> blog mapping, 0.6/0.2 weights",
         "triage_dev_select": "triage -> dev-tuned mapping, select one", "triage_tree_weighted": "triage (decision tree) -> weights",
         "oracle_label_weighted": "oracle topology label -> weights", "oracle_incident": "oracle best trio member per incident"}


def f(x, pct=True):
    return f"{100 * x:.1f}" if pct else f"{x:.3f}"


def cell(m, k="top1"):
    v = m[k]
    return f"{f(v['mean'])} [{f(v['lo'])}, {f(v['hi'])}]"


def headline():
    rows = ["| method | all | " + " | ".join(KINDS) + " | all, ties split |", "|---|" + "---|" * (len(KINDS) + 2)]
    for m in NAMES:
        if m not in M["all"]["methods"]:
            continue
        rows.append(f"| {NAMES[m]} | {cell(M['all']['methods'][m])} | "
                    + " | ".join(f(M[k]["methods"][m]["top1"]["mean"]) for k in KINDS)
                    + f" | {f(M['all']['methods'][m]['top1_ties_split']['mean'])} |")
    return "\n".join(rows)


def ties():
    rows = ["| method | " + " | ".join(f"{k}: pessimistic / ties split" for k in KINDS) + " |", "|---|" + "---|" * len(KINDS)]
    for m in ["ppr", "alarm_degree", "alarm_closeness", "alarm_degree_tb", "equal", "common_ancestor", "triage_dev_select"]:
        rows.append(f"| {NAMES[m]} | " + " | ".join(
            f"{f(M[k]['methods'][m]['top1']['mean'])} / {f(M[k]['methods'][m]['top1_ties_split']['mean'])}" for k in KINDS) + " |")
    return "\n".join(rows)


def comps():
    rows = ["| topology | comparison (A vs B) | Top-1 A - B (pp) | 95% CI | Holm p | significant |", "|---|---|---|---|---|---|"]
    for k in KINDS:
        for c in M[k]["comparisons"]:
            name = f"{NAMES[c['a']].strip('*')} vs {NAMES[c['b']].strip('*')}"
            if c["planned"]:
                ph = c["p_holm"]
                rows.append(f"| {k} | {name} | {100 * c['diff']:+.1f} | [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}] | "
                            f"{'<0.001' if ph < 1e-3 else f'{ph:.3f}'} | {'yes' if c['significant'] else 'no'} |")
            else:
                rows.append(f"| {k} | {name} *(exploratory)* | {100 * c['diff']:+.1f} | [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}] | "
                            f"unadjusted {c['p']:.3g} | n/a |")
    return "\n".join(rows)


def ablate():
    A = S["phases"]["ablate"]
    keys = [m for m in A["all"]["methods"] if "@" in m]
    base = ["equal", "alarm_degree", "ppr"]
    rows = ["| variant | " + " | ".join(KINDS) + " |", "|---|" + "---|" * len(KINDS)]
    for m in base + sorted(keys):
        rows.append(f"| `{m}` | " + " | ".join(cell(A[k]["methods"][m]) for k in KINDS) + " |")
    return "\n".join(rows)


BLOCKS = {"headline": headline, "ties": ties, "comparisons": comps, "ablate": ablate}

if __name__ == "__main__":
    if "--write" in sys.argv:
        txt = open("README.md").read()
        for name, fn in BLOCKS.items():
            txt, k = re.subn(rf"(<!-- BEGIN:{name} -->\n).*?(\n<!-- END:{name} -->)",
                             lambda mm: mm.group(1) + fn() + mm.group(2), txt, flags=re.S)
            assert k == 1, name
        open("README.md", "w").write(txt)
    else:
        for name, fn in BLOCKS.items():
            print(f"## {name}\n{fn()}\n")
