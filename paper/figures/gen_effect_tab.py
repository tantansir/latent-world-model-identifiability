"""Print LaTeX for Table 14 from the six published frozen-head JSON files.

Run from the repository root: python paper/figures/gen_effect_tab.py
The script writes only to standard output and does not edit manuscript files.
"""
import json, pathlib, numpy as np
RES = pathlib.Path(__file__).resolve().parents[2] / "exp" / "e003_motion" / "results"
def load(tag):
    return [json.load(open(f)) for f in sorted(RES.glob(f"effect_heads_VFX_s?_l0.02_{tag}_s0.json"))]
def cell3(ds, key, field, fixed=False):
    v=[d[key][field] for d in ds if key in d]
    if not v: return "---"
    if fixed: return f"{np.mean(v):.3f}"
    return f"{np.mean(v):.3f}" + (f"$\\pm${np.std(v,ddof=1):.3f}" if len(v)>1 else "$^{1}$")
def cell(ds, key, field, fixed=False):
    v=[d[key][field] for d in ds if key in d]
    if not v: return "---"
    if fixed: return f"{np.mean(v):.2f}"
    return f"{np.mean(v):.2f}" + (f"$\\pm${np.std(v,ddof=1):.2f}" if len(v)>1 else "$^{1}$")
rows=[]
for tname,tag in (("coordinate-target trunk","vr_oi_kr_mr_ac_ih"),("baseline trunk","oi_ac_ih")):
    ds=load(tag); n=len(ds)
    if not ds: continue
    rows.append(f"\\midrule\n\\rowcolor{{tabhead}} \\multicolumn{{5}}{{l}}{{{tname} ({n} seed{'s' if n>1 else ''})}} \\\\\n\\midrule")
    if tag == "vr_oi_kr_mr_ac_ih":
        rows.append(f"constant, training-mean effect (fixed) & {cell(ds,'constant_train_mean','S_delta',True)} & {cell(ds,'constant_train_mean','E_delta',True)} & {cell3(ds,'constant_train_mean','rmse',True)} & {cell(ds,'constant_train_mean','skill_vs_train_mean',True)} \\\\")
        rows.append(f"supervised control, $(s_t,\\theta,\\text{{dir}})$ (fixed) & {cell(ds,'control_true_theta','S_delta',True)} & {cell(ds,'control_true_theta','E_delta',True)} & {cell3(ds,'control_true_theta','rmse',True)} & {cell(ds,'control_true_theta','skill_vs_train_mean',True)} \\\\")
    rows.append(f"supervised control, $(s_t,\\hat\\theta,\\text{{dir}})$ & {cell(ds,'control_theta_hat','S_delta')} & {cell(ds,'control_theta_hat','E_delta')} & {cell3(ds,'control_theta_hat','rmse')} & {cell(ds,'control_theta_hat','skill_vs_train_mean')} \\\\")
    for target,lab in (("latent","latent target"),("effect","action-effect target")):
        for a in ("0","0.5","1"):
            k=f"{target}_alpha{a}"
            rows.append(f"{lab}, $\\alpha={a}$ & {cell(ds,k,'S_delta')} & {cell(ds,k,'E_delta')} & {cell3(ds,k,'rmse')} & {cell(ds,k,'skill_vs_train_mean')} \\\\")
tab="""\\begin{table}[H]
\\caption{\\textbf{Training target versus parameter precision on the mass action effect.} New heads on frozen trunks
receive the trunk state, the current object and finger state, the action window, and
$c_\\alpha=\\hat\\theta+\\alpha(\\theta-\\hat\\theta)$, and are trained with either the latent target or the
action-effect target on the same intervention samples; $\\alpha=0$ adds no test-time parameter information.
Columns: action-effect slope ratio $S^{\\Delta}$, relative error $E^{\\Delta}$, RMSE (arena units), and skill
relative to the training-mean effect, on $1{,}788$ held-out test starts; mean $\\pm$ s.d.\\ over three trunk seeds with the
head-training seed fixed. The constant and the true-parameter control do not depend on the trunk (fixed).}
\\label{tab:effect}
\\begin{center}
\\small
\\begin{tabular}{lcccc}
\\toprule
\\rowcolor{tabhead} head & $S^{\\Delta}$ & $E^{\\Delta}$ & RMSE & skill \\\\
"""+"\n".join(rows)+"""
\\bottomrule
\\end{tabular}
\\end{center}
\\end{table}
"""
print(tab, end="")
