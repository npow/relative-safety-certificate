"""Generate LaTeX tables for the paper from the experiment summaries (no hand-typed numbers)."""
import json, sys, os
R = os.path.join(os.path.dirname(__file__), "..", "..")
out = os.path.join(R, "paper")
def f(x, d=1): return f"{x:.{d}f}"
def esc(s): return s.replace("_", "\\_")

cp = json.load(open(f"{R}/experiments/v2/summaries/cartpole_summary.json")); T = cp["T"]
rows = {(r["n"], r["agent"]): r for r in cp["rows"]}
agents = [a for a in dict.fromkeys(r["agent"] for r in cp["rows"])]
L = ["\\begin{tabular}{lrrrrrrrrrr}", "\\toprule",
     " Agent & $J_A$ & $J_{\\mathrm{aware}}$ & TV & JSD & KS$^+$ & TV fin.\\ 500 & KS$^+$ EB 500 & TV fin.\\ 2000 & KS$^+$ Hoeff.\\ 2000 & KS$^+$ EB 2000 \\\\", "\\midrule"]
for a in agents:
    r5, r2 = rows[(500, a)], rows[(2000, a)]
    L.append(f"{esc(a)} & {f(r2['J_true'])} & {f(r2['aware'])} & {f(r2['plugin'])} & {f(r2['plugin_jsd'])} & {f(r2['ks_plugin'])} & {f(r5['finite'])} & {f(r5['ks_eb'])} & {f(r2['finite'])} & {f(r2['ks_finite'])} & {f(r2['ks_eb'])} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{out}/tab_cartpole.tex", "w").write("\n".join(L))

import numpy as np
SG = [("PointGoal1, seed 0", "pointgoal1_seed0"), ("PointGoal1, seed 1", "pointgoal1_seed1"), ("PointGoal1, seed 2", "pointgoal1_seed2"), ("CarGoal1, seed 0", "cargoal1_seed0")]
sgd = {lab: json.load(open(f"{R}/experiments/v2/summaries/{fn}_summary2.json")) for lab, fn in SG if os.path.exists(f"{R}/experiments/v2/summaries/{fn}_summary2.json")}
if sgd:
    def rows_of(lab, alph): return sgd[lab]["rows"][alph]
    def summ(labs, alph):
        rs = [r for lab in labs for r in rows_of(lab, alph)]
        out = dict(n=len(rs))
        out["relerr"] = float(np.median([abs(r["aware"] - r["J_true"]) / max(r["J_true"], 1.0) for r in rs]))
        def viol(key): return sum(r["J_true"] > r[key] for r in rs)
        out["tv_plug"] = viol("plugin"); out["tv_fin"] = viol("finite")
        if alph == "cost-relevant":
            for k in ("ks_plugin", "ks_finite", "top2_ks_plugin", "top2_ks_finite", "sel_ks_plugin", "sel_ks_finite"): out[k] = viol(k)
            out["sel_eb"] = sum(r["J_true"] > r["sel_eb"] for r in rs); out["all_eb"] = sum(r["J_true"] > r["all_eb"] for r in rs)
        T_ = sgd[labs[0]]["T"]
        out["fin_lt_ceiling"] = sum(r["finite"] < T_ for r in rs)
        if alph == "cost-relevant":
            out["sel_fin_lt_ceiling"] = sum(r["sel_ks_finite"] < T_ for r in rs)
            out["sel_eb_lt_ceiling"] = sum(r["sel_eb"] < T_ for r in rs); out["all_eb_lt_ceiling"] = sum(r["all_eb"] < T_ for r in rs)
            out["ratio_sel_eb"] = float(np.median([r["sel_eb"] / max(r["J_true"], 1.0) for r in rs])); out["ratio_tv_fin"] = float(np.median([r["finite"] / max(r["J_true"], 1.0) for r in rs]))
            out["sel_eb_range"] = [float(min(r["sel_eb"] for r in rs)), float(max(r["sel_eb"] for r in rs))]
            out["ratio_tv"] = float(np.median([r["plugin"] / max(r["J_true"], 1.0) for r in rs]))
            out["ratio_sel"] = float(np.median([r["sel_ks_plugin"] / max(r["J_true"], 1.0) for r in rs]))
        return out
    json.dump({lab: {a: summ([lab], a) for a in ("kmeans", "cost-relevant")} for lab in sgd} | {"all_pointgoal1": {a: summ([l for l in sgd if l.startswith("PointGoal1")], a) for a in ("kmeans", "cost-relevant")}}, open(f"{R}/experiments/v2/summaries/sg_stats.json", "w"), indent=1)
    # main table: seed 0 of PointGoal1, cost-relevant alphabet
    lab0 = "PointGoal1, seed 0"
    L = ["\\begin{tabular}{lrrrrrrrrr}", "\\toprule", " Agent & $J_A$ & $J_{\\mathrm{aware}}$ (k-means) & $J_{\\mathrm{aware}}$ (risk) & TV & KS$^+$ & KS$^+_{\\mathcal T}$ & TV fin. & KS$^+_{\\mathcal T}$ Hoeff. & KS$^+_{\\mathcal T}$ EB \\\\", "\\midrule"]
    for a, b in zip(rows_of(lab0, "kmeans"), rows_of(lab0, "cost-relevant")):
        L.append(f"{esc(a['agent'])} & {f(a['J_true'])} & {f(a['aware'])} & {f(b['aware'])} & {f(b['plugin'],0)} & {f(b['ks_plugin'],0)} & {f(b['sel_ks_plugin'],0)} & {f(b['finite'],0)} & {f(b['sel_ks_finite'],0)} & {f(b['sel_eb'],0)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{out}/tab_safetygym.tex", "w").write("\n".join(L))
    # appendix table: all tasks/seeds
    L = ["\\begin{tabular}{llrrrrrrrrr}", "\\toprule", " Task, seed & Agent & $J_B$ & $J_A$ & $J_{\\mathrm{aware}}$ (k-means) & $J_{\\mathrm{aware}}$ (risk) & TV & KS$^+_{\\mathcal T}$ & TV fin. & KS$^+_{\\mathcal T}$ Hoeff. & KS$^+_{\\mathcal T}$ EB \\\\", "\\midrule"]
    for lab in sgd:
        for a, b in zip(rows_of(lab, "kmeans"), rows_of(lab, "cost-relevant")):
            L.append(f"{lab} & {esc(a['agent'])} & {f(a['JB_true'])} & {f(a['J_true'])} & {f(a['aware'])} & {f(b['aware'])} & {f(b['plugin'],0)} & {f(b['sel_ks_plugin'],0)} & {f(b['finite'],0)} & {f(b['sel_ks_finite'],0)} & {f(b['sel_eb'],0)} \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"; L.append("\\end{tabular}")
    open(f"{out}/tab_sg_all.tex", "w").write("\n".join(L))
isj = json.load(open(f"{R}/experiments/v2/summaries/cartpole_is_summary.json"))
L = ["\\begin{tabular}{rrrrrrr}", "\\toprule", " $\\epsilon_A$ & $J_A$ & SNIS estimate & ESS (of %d) & $\\log_{10}$ max weight & TV bound & Finite-sample \\\\" % isj["n"], "\\midrule"]
for r in isj["rows"]:
    L.append(f"{r['eps_A']} & {f(r['J_true'],2)} & {f(r['snis'],2)} & {f(r['ess'])} & {f(r['log10_max_weight'])} & {f(r['ours_plugin'])} & {f(r['ours_finite'])} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{out}/tab_is.tex", "w").write("\n".join(L))
eb_by_eps = {r["eps_A"]: r["is_eb_upper"] for r in isj["rows"]}
def sci(x):
    exponent = int(np.floor(np.log10(x)))
    return f"{x / (10 ** exponent):.1f}\\times10^{{{exponent}}}"
L = ["\\begin{tabular}{rrrrrrrr}", "\\toprule", " $\\epsilon_A$ & $J_A$ & SNIS estimate & ESS (of %d) & $\\log_{10}$ max weight & IS EB upper & TV bound & Finite-sample \\\\" % isj["n"], "\\midrule"]
for r in isj["rows"]:
    row = f"{r['eps_A']} & {f(r['J_true'],2)} & {f(r['snis'],2)} & {f(r['ess'])} & {f(r['log10_max_weight'])} & ${sci(eb_by_eps[r['eps_A']])}$ & {f(r['ours_plugin'])} & {f(r['ours_finite'])} "
    L.append(row + chr(92) * 2)
L += ["\\bottomrule", "\\end{tabular}"]
open(f"{out}/tab_is.tex", "w").write("\n".join(L))
print("ok")
