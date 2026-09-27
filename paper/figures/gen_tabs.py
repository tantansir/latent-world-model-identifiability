"""Print LaTeX for Tables 10-12 from the published per-seed result files.

Run from the repository root: python paper/figures/gen_tabs.py
Release slopes and action-effect errors come from intervention_metrics.json;
summarize_interventions.py regenerates that file from the simulator outputs.
"""
import json, pathlib, numpy as np
RES = pathlib.Path(__file__).resolve().parents[2] / "exp" / "e003_motion" / "results"
METRICS = json.loads((RES / "intervention_metrics.json").read_text(encoding="utf-8"))["metrics"]
def J(f):
    try: return json.loads((RES / f).read_text(encoding="utf-8"))
    except FileNotFoundError: return None
def fmt(v): return "---" if v is None or (isinstance(v,float) and np.isnan(v)) else f"{v:.2f}"
def fmtn(vals):
    v=[x for x in vals if x is not None]
    if not v: return "---"
    return f"{np.mean(v):.2f}"+("$^{1}$" if len(v)==1 else "")
def seeds(tag, getter, suf=""):
    out=[]
    for s in (0,1,2):
        d=J(f"{tag.replace('S',str(s))}{suf}.json")
        try: out.append(None if d is None else getter(d))
        except (KeyError, TypeError): out.append(None)
    return out
def cell(vals):
    v=[x for x in vals if x is not None]
    if len(v)==3: return "/".join(f"{x:.2f}" for x in v)
    if len(v)==1: return f"{v[0]:.2f}$^{{1}}$"
    return "---"
def mean(vals):
    v=[x for x in vals if x is not None]; return None if not v else float(np.mean(v))
def rel(tag,s):
    key = f"func_cf_{tag.replace('S',str(s))}_release"
    return METRICS[key]["drag_release_slope_ratio"]

def eD(tag,s):
    key = f"func_cf_{tag.replace('S',str(s))}"
    return METRICS[key]["mass_action_effect_relative_error"]

models=[("additive","VFX_sS_l0.02_ac"),("interaction, no state","VFX_sS_l0.02_ac_ih"),
        ("baseline","VFX_sS_l0.02_oi_ac_ih"),("coordinate","VFX_sS_l0.02_vr_oi_kr_mr_ac_ih"),
        ("+ self-cond.","VFX_sS_l0.02_vr_oi_kr_mr_sc_ac_ih"),("direct sup.","VFX_sS_l0.02_sid_ac_ih"),
        ("oracle","VFX_sS_l0.02_or_oi_ac_ih")]
LEGEND="Models: additive = VFX with the additive head and no state input; interaction, no state = interaction head without state input; baseline, coordinate, oracle = the three models of Table~\\ref{tab:three} (oracle: true parameters); + self-cond.\\ = coordinate targets with the model's own coordinate estimates fed to the predictor; direct sup.\\ = parameter-supervision heads, no state input."
rowsA=[]; rowsB=[]; rowsC=[]; kstart={}
for name,tag in models:
    push=seeds("func_cf_"+tag, lambda d:d["m_partial_d16"]["ratio"]); ae=seeds("func_cf_"+tag, lambda d:d["m_partial_d16_action_effect"]["ratio"])
    ed=[eD(tag,s) for s in (0,1,2)]; nat=seeds("func_nat_"+tag, lambda d:d["state + future actions"]["ratio"]); rl=[rel(tag,s) for s in (0,1,2)]
    rowsA.append(f"{name} & {cell(push)} & {cell(ae)} & {cell(ed)} & {cell(nat)} & {cell(rl)} \\\\")
    kp=seeds("func_impact_"+tag, lambda d:d["push"]["S"]); kd=seeds("func_impact_"+tag, lambda d:d["delta"]["S"]); ke=seeds("func_impact_"+tag, lambda d:d["delta"]["E_delta"]); kn=seeds("func_impact_"+tag, lambda d:d["natural"]["S"])
    kr=seeds("func_impact_"+tag, lambda d:d["probe_at_intervention"]["log_k"])
    if any(x is not None for x in kp): rowsB.append(f"{name} & {cell(kp)} & {cell(kd)} & {cell(ke)} & {cell(kn)} & {cell(kr)} \\\\")
    sm=seeds("func_cf_"+tag, lambda d:d["probe_at_intervention"]["log_m"]); sg=seeds("func_cf_"+tag, lambda d:d["probe_at_intervention"]["gamma"], "_release"); sk=kr; kstart[tag]=mean(kr)
    cm=seeds(tag, lambda d:d["probe"]["trunk"]["contact"]["log_mass"]); cg=seeds(tag, lambda d:d["probe"]["trunk"]["contact"]["gamma"]); ck=seeds(tag, lambda d:d["probe"]["trunk"]["contact"]["log_stiffness"]); gg=seeds(tag, lambda d:d["probe"]["trunk"]["glide"]["gamma"])
    rowsC.append(f"{name} & {fmtn(sm)} & {fmtn(sg)} & {fmtn(sk)} & {fmtn(cm)} & {fmtn(cg)} & {fmtn(ck)} & {fmtn(gg)} \\\\")
tabs="""\\begin{table}[H]
\\caption{\\textbf{Mass and drag tests, per seed.} Every slash separates the
three representation seeds. Columns: mass push-branch
slope ratio, mass action-effect ratio $S^{\\Delta}$ and relative error
$E^{\\Delta}$, mass continuation ratio $S^{\\mathrm{nat}}$, and drag
release-branch slope ratio. """+LEGEND+"""}
\\label{tab:threeseeds}
\\begin{center}
\\scriptsize
\\setlength{\\tabcolsep}{3pt}
\\begin{tabular}{lccccc}
\\toprule
\\rowcolor{tabhead} model & $m$ push & $m$ $S^{\\Delta}$ & $m$ $E^{\\Delta}$ & $m$ $S^{\\mathrm{nat}}$ & $\\gamma$ release \\\\
\\midrule
"""+"\n".join(rowsA)+"""
\\bottomrule
\\end{tabular}
\\end{center}
\\end{table}

\\begin{table}[H]
\\caption{\\textbf{Stiffness test, per seed} (slashes separate seeds). Push-branch slope ratio, action-effect ratio $S^{\\Delta}$ and
relative error $E^{\\Delta}$ (push minus retreat), continuation ratio
$S^{\\mathrm{nat}}$, and the stiffness readout at the impact start.}
\\label{tab:kseeds}
\\begin{center}
\\scriptsize
\\begin{tabular}{lccccc}
\\toprule
\\rowcolor{tabhead} model & $k$ push & $k$ $S^{\\Delta}$ & $k$ $E^{\\Delta}$ & $k$ $S^{\\mathrm{nat}}$ & $k$ readout at start \\\\
\\midrule
"""+"\n".join(rowsB)+"""
\\bottomrule
\\end{tabular}
\\end{center}
\\end{table}

\\begin{table}[H]
\\caption{\\textbf{Readouts of the same checkpoints} (three-seed means). Readout at the
intervention start is the episode-grouped ridge $R^2$ on the mass-push
contexts ($m$), the release contexts ($\\gamma$), and the impact contexts
($k$); contact and glide columns are the ridge probes of Table~\\ref{tab:factorial}
on the standard probe windows. The oracle is given the parameters.}
\\label{tab:threeread}
\\begin{center}
\\scriptsize
\\begin{tabular}{lccccccc}
\\toprule
\\rowcolor{tabhead} model & start $m$ & start $\\gamma$ & start $k$ & contact $m$ & contact $\\gamma$ & contact $k$ & glide $\\gamma$ \\\\
\\midrule
"""+"\n".join(rowsC)+"""
\\bottomrule
\\end{tabular}
\\end{center}
\\end{table}
"""
print(tabs, end="")
