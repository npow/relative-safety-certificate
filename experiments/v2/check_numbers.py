"""Recompute every number quoted in the paper text from the summary files; exit 1 on mismatch."""
import json, os, sys
import numpy as np
S = os.path.join(os.path.dirname(__file__), "summaries")
L = lambda n: json.load(open(f"{S}/{n}"))
ok = True
def chk(name, got, want):
    global ok
    good = got == want; ok &= good
    print(("OK  " if good else "FAIL"), name, got, "" if good else f"(text says {want})")
rr = lambda x, d=0: round(float(x), d) if d else int(round(float(x)))

# ---------------- CartPole ----------------
cp = L("cartpole_summary.json"); T = cp["T"]; rows = cp["rows"]
R = {(r["n"], r["agent"]): r for r in rows}; ag = sorted({r["agent"] for r in rows})
chk("cartpole cells", len(rows), 27); chk("cartpole evaluations", sum(r["reps"] for r in rows), 405)
tot = lambda k: sum(r[k] * r["reps"] for r in rows)
for k in ("viol_plugin", "viol_plugin_jsd", "viol_finite", "viol_ks_finite", "viol_ks_eb"): chk("cartpole violations " + k, round(tot(k)), 0)
chk("cartpole KS plug-in violations", round(tot("viol_ks_plugin")), 11)
chk("KS plug-in violations: all agents near baseline (|J_A-J_B|<=0.2)", all(abs(r["J_true"] - r["JB_true"]) <= 0.2 for r in rows if r["viol_ks_plugin"] > 0), True)
chk("JB cartpole", rr(R[(2000, "eps0.1")]["JB_true"], 1), 0.1)
for n, c in ((100, 5), (500, 8), (2000, 9)): chk(f"TV finite < ceiling n={n}", sum(R[(n, a)]["finite"] < T for a in ag), c)
for n, c, lo, hi in ((100, 8, 84.5, 232.9), (500, 9, 16.9, 155.9), (2000, 9, 4.3, 139.7)):
    eb = [R[(n, a)]["ks_eb"] for a in ag]
    chk(f"EB < ceiling n={n}", sum(x < T for x in eb), c); chk(f"EB range n={n}", (rr(min(eb), 1), rr(max(eb), 1)), (lo, hi))
near = ("eps0.02", "eps0.05", "eps0.1", "gain0.1", "gain0.25", "lazy")
chk("near-baseline agents have J_A<0.8", all(R[(2000, a)]["J_true"] < 0.8 for a in near), True)
chk("EB at 2000 near-baseline range", (rr(min(R[(2000, a)]["ks_eb"] for a in near), 1), rr(max(R[(2000, a)]["ks_eb"] for a in near), 1)), (4.3, 8.8))
rt = [R[(2000, a)]["finite"] / R[(2000, a)]["ks_eb"] for a in near]; chk("TV fin / EB range", (rr(min(rt)), rr(max(rt))), (5, 11))
chk("aware within 0.4 at n=2000", max(abs(R[(2000, a)]["aware"] - R[(2000, a)]["J_true"]) for a in ag) < 0.4, True)
tvr = lambda a: R[(2000, a)]["plugin"] / R[(2000, a)]["J_true"]; ksr = lambda a: R[(2000, a)]["ks_plugin"] / R[(2000, a)]["J_true"]
noise = ("eps0.02", "eps0.05", "eps0.1", "eps0.2")
chk("TV/J noise eps<=0.2", (rr(min(map(tvr, noise))), rr(max(map(tvr, noise)))), (6, 18)); chk("TV/J lazy ~60", rr(tvr("lazy")), 60)
chk("KS/J noise eps<=0.2", (rr(min(map(ksr, noise))), rr(max(map(ksr, noise)))), (3, 6)); chk("KS/J lazy", rr(ksr("lazy"), 1), 3.6)

# ---------------- importance sampling ----------------
IS = L("cartpole_is_summary.json")["rows"]
chk("IS J_B", rr(IS[0]["J_B"], 1), 20.0)
chk("IS ESS", tuple(rr(r["ess"], 1) for r in IS), (2.2, 3.5, 65.8))
chk("IS SNIS errors", tuple(rr(abs(r["snis"] - r["J_true"]), 2) for r in IS), (0.57, 0.33, 0.45))
chk("IS log10 max weight range", (rr(min(r["log10_max_weight"] for r in IS)), rr(max(r["log10_max_weight"] for r in IS))), (5, 12))
chk("IS log10 Hoeffding slack range", (rr(min(r["log10_hoeffding_slack"] for r in IS), 1), rr(max(r["log10_hoeffding_slack"] for r in IS), 1)), (5.7, 12.7))
chk("IS plug-in TV range", (rr(min(r["ours_plugin"] for r in IS), 1), rr(max(r["ours_plugin"] for r in IS), 1)), (53.3, 79.6))
chk("IS ours finite range", (rr(min(r["ours_finite"] for r in IS), 1), rr(max(r["ours_finite"] for r in IS), 1)), (92.9, 119.1))
chk("IS ours plugin > J_B", all(r["ours_plugin"] > r["J_B"] for r in IS), True)
chk("IS ours finite all > J_B", all(r["ours_finite"] > r["J_B"] for r in IS), True)

# ---------------- Safety-Gymnasium ----------------
files = ["pointgoal1_seed0", "pointgoal1_seed1", "pointgoal1_seed2", "cargoal1_seed0"]
D = {f: L(f"{f}_summary2.json") for f in files}; AL = {f: L(f"{f}_alphabet.json") for f in files}; ON = {f: L(f"{f}_onset.json") for f in files}
chk("JB per seed", tuple(rr(D[f]["rows"]["kmeans"][0]["JB_true"], 1) for f in files), (3.7, 82.4, 6.7, 58.7))
chk("windows", tuple(AL[f]["win"] for f in files), (28, 28, 28, 40)); chk("pilot R2", tuple(rr(AL[f]["r2"], 2) for f in files), (0.77, 0.81, 0.93, 0.82))
chk("onset sets", tuple(tuple(ON[f]["onset"]) for f in files), ((6,), (1, 2), (4, 5), (2, 3)))
cells = [(f, a, b) for f in files for a, b in zip(D[f]["rows"]["kmeans"], D[f]["rows"]["cost-relevant"])]
chk("cells", len(cells), 32)
pg = [c for c in cells if c[0].startswith("pointgoal")]; cg = [c for c in cells if c[0].startswith("cargoal")]
rel = lambda r: abs(r["aware"] - r["J_true"]) / max(r["J_true"], 1.0)
chk("rel err kmeans PG", rr(np.median([rel(a) for _, a, _ in pg]), 2), 0.38); chk("rel err cost-rel PG", rr(np.median([rel(b) for _, _, b in pg]), 2), 0.17)
chk("rel err kmeans CG", rr(np.median([rel(a) for _, a, _ in cg]), 2), 0.06); chk("rel err cost-rel CG", rr(np.median([rel(b) for _, _, b in cg]), 2), 0.15)
s0 = [c for c in cells if c[0] == "pointgoal1_seed0"][:4]
chk("seed0 aware kmeans lam range", (rr(min(a["aware"] for _, a, _ in s0), 1), rr(max(a["aware"] for _, a, _ in s0), 1)), (9.8, 10.8))
chk("seed0 aware cost-rel lam range", (rr(min(b["aware"] for _, _, b in s0)), rr(max(b["aware"] for _, _, b in s0))), (83, 117))
chk("seed0 J lam range", (rr(min(a["J_true"] for _, a, _ in s0)), rr(max(a["J_true"] for _, a, _ in s0))), (73, 115))
v = lambda sel, key: sum(r["J_true"] > r[key] for _, a, b in cells for r in [sel(a, b)])
chk("finite violations (TV, Hoeff, EB, all-thr EB)", (v(lambda a, b: b, "finite"), v(lambda a, b: b, "ks_finite"), v(lambda a, b: b, "sel_ks_finite"), v(lambda a, b: b, "sel_eb"), v(lambda a, b: b, "all_eb")), (0, 0, 0, 0, 0))
chk("kmeans TV finite violations", v(lambda a, b: a, "finite"), 0)
chk("sel plug-in violations (eta>0)", v(lambda a, b: b, "sel_ks_plugin"), 3); chk("sel plug-in violations (eta=0)", v(lambda a, b: b, "sel0_ks_plugin"), 6); chk("kmeans TV plug-in violations", v(lambda a, b: a, "plugin"), 1)
chk("cost-rel TV plug-in violations", v(lambda a, b: b, "plugin"), 0); chk("fixed {5,6} plug-in violations", v(lambda a, b: b, "top2_ks_plugin"), 14)
chk("kmeans TV plug-in violation is CarGoal1", [c[0] for c in cells if c[1]["J_true"] > c[1]["plugin"]], ["cargoal1_seed0"])
eb = [b["sel_eb"] for _, _, b in cells]; chk("EB below 1000", sum(x < 1000 for x in eb), 32); chk("EB range", (rr(min(eb)), rr(max(eb))), (137, 406))
chk("TV fin below 1000 (cost-rel)", sum(b["finite"] < 1000 for _, _, b in cells), 30)
med = lambda cs, f: float(np.median([f(b) / max(b["J_true"], 1.0) for _, _, b in cs]))
chk("plug-in sel ratio PG, CG", (rr(med(pg, lambda b: b["sel_ks_plugin"]), 1), rr(med(cg, lambda b: b["sel_ks_plugin"]), 1)), (1.4, 1.2))
chk("EB ratio PG, CG", (rr(med(pg, lambda b: b["sel_eb"]), 1), rr(med(cg, lambda b: b["sel_eb"]), 1)), (6.3, 4.1))
chk("TV fin ratio PG, CG", (rr(med(pg, lambda b: b["finite"]), 1), rr(med(cg, lambda b: b["finite"]), 1)), (15.1, 7.7))
chk("all EB below 1000", sum(b["all_eb"] < 1000 for _, _, b in cells), 32)
chk("seed1 certificate-pool baseline mean ~64", rr(min(b["sel0_ks_plugin"] for f, _, b in cells if f == "pointgoal1_seed1")), 64)
chk("abstract: EB median ratio about 4 to 6 (per-task medians)", (rr(med(cg, lambda b: b["sel_eb"])), rr(med(pg, lambda b: b["sel_eb"]))), (4, 6))
chk("eta_bar values", sorted({round(b["eta_bar"], 3) for _, _, b in cells}), [0.05, 0.055])
chk("monotone: agents non-monotone cells", sum(not b["mono_agent"] for _, _, b in cells), 1); chk("monotone: baseline non-monotone cells", sum(not b["mono_base"] for _, _, b in cells), 0)
chk("non-monotone agent is seed1 lam1.0", [(f, a["agent"]) for f, a, b in cells if not b["mono_agent"]], [("pointgoal1_seed1", "lam1.0")])

# noise, drift, pilot check
bl = [D[f]["baseline"] for f in files]
chk("baseline |z| max", rr(max(abs(b["z"]) for b in bl), 1), 1.1); chk("baseline truth SD max", rr(max(b["sd_truth"] for b in bl)), 227)
chk("J_se range", (rr(min(b["J_se"] for _, _, b in cells), 1), rr(max(b["J_se"] for _, _, b in cells), 1)), (1.3, 12.7))
for key in ("sel_ks_plugin", "sel0_ks_plugin", "plugin", "sel_eb", "finite"): chk("2SE-definite violations " + key, sum(b["J_true"] - 2 * b["J_se"] > b[key] for _, _, b in cells), 0)
chk("EB minus plug-in slack range", (rr(min(b["sel_eb"] - b["sel_ks_plugin"] for _, _, b in cells)), rr(max(b["sel_eb"] - b["sel_ks_plugin"] for _, _, b in cells))), (127, 258))
chk("drift median relative", rr(np.median([abs(b["drift"]) / max(b["J_true"], 1.0) for _, _, b in cells]), 2), 0.02)
chk("drift max abs", rr(max(abs(b["drift"]) for _, _, b in cells), 1), 18.0); chk("drift positive cells", sum(b["drift"] > 0 for _, _, b in cells), 19)
chk("drift max cell is seed1", max(cells, key=lambda c: abs(c[2]["drift"]))[0], "pointgoal1_seed1"); chk("drift unseen mass", max(b["drift_unseen_mass"] for _, _, b in cells), 0.0)
def pil(f):
    o = ON[f]; d = [abs(x - y) for x, y, a, b in zip(o["rates_sigma0"], o["rates_sigma15"], o["visits_sigma0"], o["visits_sigma15"]) if x is not None and y is not None and a >= 20 and b >= 20]
    return rr(max(d), 2)
chk("pilot rate differences", tuple(pil(f) for f in files), (0.0, 0.31, 0.06, 0.27))

# ---------------- every quoted figure must appear in the paper / abstract ----------------
tex = open(os.path.join(os.path.dirname(__file__), "..", "..", "paper", "main.tex")).read()
ab = open(os.path.join(os.path.dirname(__file__), "..", "..", "publishfun", "abstract.txt")).read()
for s in ["405 evaluations", "4.3--8.8", "5 to 11 times", "11 of 405", "84.5--232.9", "16.9--155.9", "4.3--139.7", "within 0.4", "6--18", "3--6", "3.6",
          "0.57, 0.33, 0.45", "2.2, 3.5 and 65.8", "$10^{5}$ and $10^{12}$", "$10^{5.7}$ and $10^{12.7}$", "53.3--79.6", "92.9--119.1",
          "$J_B=3.7$, $82.4$ and $6.7$", "0.77,0.81,0.93", "0.38", "0.17", "0.06", "0.15", "9.8--10.8", "73--115", "83--117",
          "in 3 of 32 cells (6 of 32 with $\\eta=0$)", "14 of the 32", "137--406", "30 of 32", "1.4 (PointGoal1) and 1.2 (CarGoal1)",
          "6.3 (PointGoal1) and 4.1 (CarGoal1)", "15.1 and 7.7", "31 of 32", "$\\eta=0.05$ (0.055 for seed 2)", "1 of 32 (CarGoal1)", "127--258", "1.3--12.7", "$|z|\\le1.1$", "227 for a mean of 82", "2\\% of $\\max\\{J_A,1\\}$", "18.0 cost units", "19 of 32 cells", "0.31 and 0.27", "0.06 for seed 2"]:
    chk("tex contains: " + s, s in tex, True)
for s in ["405 evaluations", "4.3–8.8", "5 to 11 times", "3 of 32", "4 to 6 times", "range-based importance-sampling"]:
    chk("abstract contains: " + s, s in ab, True)
sys.exit(0 if ok else 1)
