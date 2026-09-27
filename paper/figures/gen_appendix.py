"""Print LaTeX for Table 2 and Tables 6-9 from published result JSON.

Run from the repository root: python paper/figures/gen_appendix.py
The script writes only to standard output and does not edit manuscript files.
"""
import json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "exp" / "e003_motion" / "results"


def J(tag):
    f = RES / f"{tag}.json"
    return json.load(open(f)) if f.exists() else None


def get(d, *path):
    for k in path:
        if d is None or k not in d:
            return None
        d = d[k]
    return d


def f2(x):
    return "---" if x is None else f"${x:.2f}$"


def ms(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return "---"
    if len(v) == 1:
        return f"${v[0]:.2f}$"
    return f"${np.mean(v):.2f}\\pm{np.std(v, ddof=1):.2f}$"


NAMES = {"V": "V", "VF": "VF", "VXp": "VX$_p$", "VXt": "VX$_t$", "VX": "VX", "VFX": "VFX"}
out = []

# ---------------------------------------------------------------- MLP probes (corrected factorial, 3 seeds)
rows = []
for v in NAMES:
    cells = [ms([get(J(f"{v}_s{s}_l0.02_ac"), "probe", "trunk_mlp", "contact", k) for s in (0, 1, 2)])
             for k in ("log_mass", "gamma", "log_stiffness")]
    rows.append(f"{NAMES[v]} & " + " & ".join(cells) + " \\\\")
out.append(r"""\paragraph{Nonlinear probes.} Early-stopped MLP probes on the same contact
windows reproduce the ridge results of Table~\ref{tab:factorial}
(Table~\ref{tab:mlp}).

\begin{table}[H]
\caption{\textbf{MLP probes.} Early-stopped two-layer MLP readout ($R^2$, mean
$\pm$ s.d.\ over three seeds) from the frozen predictor state on contact
windows.}
\label{tab:mlp}
\begin{center}
\small
\begin{tabular}{lccc}
\toprule
\rowcolor{tabhead} Model & $\log m$ & $\gamma$ & $\log k$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{center}
\end{table}
""")

# ---------------------------------------------------------------- objective controls (seed 0)
def ctrl_row(name, tag):
    d = J(tag)
    c = [get(d, "probe", "trunk", "contact", k) for k in ("log_mass", "gamma", "log_stiffness")]
    return f"{name} & " + " & ".join(f2(x) for x in c) + f" & {f2(get(d, 'probe_r2_pos_frame'))} & {f2(get(d, 'probe', 'velocity_r2', 'moving'))} \\\\"


crow = [ctrl_row("VFX", "VFX_s0_l0.02_ac"),
        ctrl_row("VFX + parameter supervision", "VFX_s0_l0.02_sid_ac"),
        ctrl_row("VFX, Gaussian NLL", "VFX_s0_l0.02_nll_ac"),
        ctrl_row("VFX, pixel reconstruction", "VFX_s0_l0.02_rc_ac"),
        ctrl_row("V, pixel reconstruction", "V_s0_l0.02_rc_ac")]
out.append(r"""\paragraph{Objective controls.} Keeping the VFX predictor, data, and
probes fixed, we change only the objective (Table~\ref{tab:controls}).
Supervising the parameters directly raises every parameter readout, most for
stiffness and mass; pixel reconstruction in place of latent prediction leaves
drag readout low, and stiffness remains decodable with touch forecasting.

\begin{table}[H]
\caption{\textbf{Objective controls} (one seed). Ridge readout from the
predictor state on contact windows; position from the frame embedding;
velocity on moving windows.}
\label{tab:controls}
\begin{center}
\small
\begin{tabular}{lccccc}
\toprule
\rowcolor{tabhead} Objective & $\log m$ & $\gamma$ & $\log k$ & position & velocity \\
\midrule
""" + "\n".join(crow) + r"""
\bottomrule
\end{tabular}
\end{center}
\end{table}
""")

# ---------------------------------------------------------------- SIGReg weight sweep (VFX, seed 0)
lams = ["0.3", "0.1", "0.02", "0.01", "0.005"]
tags = {l: (f"VFX_s0_l{l}_ac" if l != "0.1" else "VFX_s0_ac") for l in lams}   # lambda=0.1 is the default and carries no suffix
def lrow(name, fn):
    return name + " & " + " & ".join(f2(fn(J(tags[l]))) for l in lams) + " \\\\"
lrows = [lrow("$\\log k$", lambda d: get(d, "probe", "trunk", "contact", "log_stiffness")),
         lrow("$\\log m$", lambda d: get(d, "probe", "trunk", "contact", "log_mass")),
         lrow("$\\gamma$", lambda d: get(d, "probe", "trunk", "contact", "gamma")),
         lrow("velocity", lambda d: get(d, "probe", "velocity_r2", "moving"))]
out.append(r"""\paragraph{Regularizer weight.} A lighter SIGReg weight sharpens every
readout of the VFX model (Table~\ref{tab:lambda}).

\begin{table}[H]
\caption{\textbf{SIGReg weight} (VFX, one seed). Ridge readout on contact
windows; velocity on moving windows.}
\label{tab:lambda}
\begin{center}
\small
\begin{tabular}{lccccc}
\toprule
\rowcolor{tabhead} $\lambda$ & """ + " & ".join(f"${l}$" for l in lams) + r""" \\
\midrule
""" + "\n".join(lrows) + r"""
\bottomrule
\end{tabular}
\end{center}
\end{table}
""")

# ---------------------------------------------------------------- touch-loss contact weight (VFX, seed 0)
tws = [("4", "VFX_s0_l0.02_ac"), ("1", "VFX_s0_l0.02_tw1_ac"), ("0.25", "VFX_s0_l0.02_tw0.25_ac"), ("0", "VFX_s0_l0.02_tw0_ac")]
trows = []
for name, key in (("$\\log k$", "log_stiffness"), ("$\\log m$", "log_mass"), ("$\\gamma$", "gamma")):
    trows.append(name + " & " + " & ".join(f2(get(J(t), "probe", "trunk", "contact", key)) for _, t in tws) + " \\\\")
out.append(r"""\paragraph{Touch-loss weight on contact frames.} Stiffness readout depends on
how strongly the touch loss weighs contact frames, while mass and drag barely
move (Table~\ref{tab:touchw}).

\begin{table}[H]
\caption{\textbf{Contact-frame weight of the touch loss} (VFX, one seed; the
default is 4). Ridge readout on contact windows.}
\label{tab:touchw}
\begin{center}
\small
\begin{tabular}{lcccc}
\toprule
\rowcolor{tabhead} weight & """ + " & ".join(f"${w}$" for w, _ in tws) + r""" \\
\midrule
""" + "\n".join(trows) + r"""
\bottomrule
\end{tabular}
\end{center}
\end{table}
""")

# ---------------------------------------------------------------- prediction horizon (V, three seeds) -> main-text table
HZ = [("one next-step head", "D1"), ("one next-step head, given $a_{t+1}$", "D1_e1x1_np"),
      ("three next-step heads, given $a_{t+1}$", "D1_e1x3_np"), ("heads at $\\Delta\\in\\{1,4,16\\}$", "")]
REG = [("V_s{s}_nd_ac", True), ("V_s{s}_l0.02_nd_ac", True), ("V_s{s}_l0.02_ac", False)]   # lambda=0.1 is the default and carries no suffix
hrows = []
for name, suf in HZ:
    cells = []
    for pre, has_e in REG:
        htags = [pre.format(s=s) + ("_" + suf if suf else "") for s in (0, 1, 2)]
        cells.append(ms([get(J(t), "probe_r2_pos_frame") for t in htags]))
        cells.append(ms([get(J(t), "probe", "velocity_r2", "moving") for t in htags]))
    hrows.append(name + " & " + " & ".join(cells) + " \\\\")
HORIZON_TABLE = r"""\begin{table}[htb]
\caption{\textbf{Multi-horizon prediction improves position and velocity readout.}
Within each observation and regularization setting, vision-only V models
differ only in their latent prediction heads (three seeds). Object position
is read from the frame embedding and velocity
from the predictor state; $a_{t+1}$ is the action that produces the next frame.
The three-head and multi-horizon rows share a summed loss weight of $2$.}
\label{tab:horizon}
\begin{center}
\small
\resizebox{\linewidth}{!}{\begin{tabular}{lcccccc}
\toprule
\rowcolor{tabhead} & \multicolumn{2}{c}{grayscale, $\lambda=0.1$} & \multicolumn{2}{c}{grayscale, $\lambda=0.02$} & \multicolumn{2}{c}{frame + difference, $\lambda=0.02$} \\
\rowcolor{tabhead} Latent prediction heads & position & velocity & position & velocity & position & velocity \\
\midrule
""" + "\n".join(hrows) + r"""
\bottomrule
\end{tabular}}
\end{center}
\end{table}
"""

block = "\n".join(out)
print(HORIZON_TABLE + "\n" + block, end="")
