"""Experiment grid. Twins are seeded by (kind, n, seed); incidents by (seed, condition, i). Dev seeds (1-5) are
used only for pilots and triage calibration; evaluation seeds are never looked at before the grid.

Round-1 review: the main and p_link phases use 30 eval twins per topology (seeds 100-129) so that the planned
comparisons can be tested at the twin level; poll / ablate keep the first 10, size the first 5."""
import itertools

KINDS = ("hierarchical", "star", "mesh", "hybrid")
DEV_SEEDS = tuple(range(1, 6))
EVAL_SEEDS = tuple(range(100, 130))
EVAL10 = EVAL_SEEDS[:10]
P_LINK = (0.3, 0.5, 0.7, 0.9)
NOISE = (0.0, 0.25, 0.5, 1.0)
SILENT = (0.0, 0.5, 1.0)
DEFAULT = dict(noise=0.25, p_silent=0.5, poll_interval=60.0)


def phases():
    """phase -> list of (kind, n, seed, [conditions], incidents_per_condition, ablations)."""
    main = [dict(noise=z, p_silent=s, poll_interval=60.0) for z, s in itertools.product(NOISE, SILENT)]
    poll = [dict(noise=0.25, p_silent=s, poll_interval=p) for p in (0.0, 15.0, 60.0, 300.0) for s in (0.0, 0.5)]
    poll += [dict(noise=0.0, p_silent=0.0, poll_interval=0.0, jitter=0.0)]  # ideal clocks: temporal baseline's best case
    # review M1: neighbour LINK_DOWN reliability (what alarm-degree counts), plus an upstream-only symptom model
    plink = [dict(noise=0.25, p_silent=s, poll_interval=60.0, p_link=p) for p in P_LINK for s in (0.0, 0.5)]
    plink += [dict(noise=0.25, p_silent=s, poll_interval=60.0, p_link=0.9, link_model="upstream") for s in (0.0, 0.5)]
    return {
        "dev": [(k, 1000, s, main, 30, False) for k in KINDS for s in DEV_SEEDS],
        "main": [(k, 1000, s, main, 60, False) for k in KINDS for s in EVAL_SEEDS],
        "plink": [(k, 1000, s, plink, 40, False) for k in KINDS for s in EVAL_SEEDS],
        "poll": [(k, 1000, s, poll, 60, False) for k in KINDS for s in EVAL10],
        "ablate": [(k, 1000, s, [DEFAULT], 100, True) for k in KINDS for s in EVAL10],
        "size": [(k, n, s, [DEFAULT], 40, False) for k in KINDS for n in (500, 2000, 5000) for s in EVAL_SEEDS[:5]],
    }


def cond_id(c):
    extra = "".join(f"_{k}{v:g}" if not isinstance(v, str) else f"_{k}-{v}"
                    for k, v in sorted(c.items()) if k not in ("noise", "p_silent", "poll_interval"))
    return f"nz{c['noise']:g}_si{c['p_silent']:g}_po{c['poll_interval']:g}{extra}"
