"""Experiment grid. Twins are seeded by (kind, n, seed); incidents by (seed, condition, i). Dev seeds (1-5) are
used only for pilots and triage calibration; evaluation seeds (100-109) are never looked at before the grid."""
import itertools

KINDS = ("hierarchical", "star", "mesh", "hybrid")
DEV_SEEDS = tuple(range(1, 6))
EVAL_SEEDS = tuple(range(100, 110))
NOISE = (0.0, 0.25, 0.5, 1.0)
SILENT = (0.0, 0.5, 1.0)
DEFAULT = dict(noise=0.25, p_silent=0.5, poll_interval=60.0)


def phases():
    """phase -> list of (kind, n, seed, [conditions], incidents_per_condition, ablations)."""
    main = [dict(noise=z, p_silent=s, poll_interval=60.0) for z, s in itertools.product(NOISE, SILENT)]
    poll = [dict(noise=0.25, p_silent=s, poll_interval=p) for p in (0.0, 15.0, 60.0, 300.0) for s in (0.0, 0.5)]
    poll += [dict(noise=0.0, p_silent=0.0, poll_interval=0.0, jitter=0.0)]  # ideal clocks: temporal baseline's best case
    return {
        "dev": [(k, 1000, s, main, 30, False) for k in KINDS for s in DEV_SEEDS],
        "main": [(k, 1000, s, main, 60, False) for k in KINDS for s in EVAL_SEEDS],
        "poll": [(k, 1000, s, poll, 60, False) for k in KINDS for s in EVAL_SEEDS],
        "ablate": [(k, 1000, s, [DEFAULT], 100, True) for k in KINDS for s in EVAL_SEEDS],
        "size": [(k, n, s, [DEFAULT], 40, False) for k in KINDS for n in (500, 2000, 5000) for s in EVAL_SEEDS[:5]],
    }


def cond_id(c):
    extra = "".join(f"_{k}{v:g}" for k, v in sorted(c.items()) if k not in ("noise", "p_silent", "poll_interval"))
    return f"nz{c['noise']:g}_si{c['p_silent']:g}_po{c['poll_interval']:g}{extra}"
