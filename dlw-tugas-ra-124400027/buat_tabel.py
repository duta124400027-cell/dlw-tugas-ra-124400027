"""Membuat tabel LaTeX (rerata ± sd, 3 seed) dan grafik laporan dari hasil/ablasi.csv.
Pakai:  python buat_tabel.py            (dari akar repo)
Keluaran: hasil/tabel/*.tex, hasil/fig_A.png, hasil/fig_B.png, hasil/ringkasan.txt
Semua angka berasal langsung dari CSV; tidak ada angka yang diketik manual."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("hasil/ablasi.csv")
OUT = CSV.parent / "tabel"; OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["ANN", "RNN", "LSTM", "GRU"]
WARNA = {"ANN": "#D4A017", "RNN": "#C8102E", "LSTM": "#0B1F3A", "GRU": "#1E8C8C"}
df = pd.read_csv(CSV, dtype={"nilai": str, "kunci": str})
KAL = r"\multicolumn{5}{c}{\textit{(bagian ini belum dijalankan)}}"

def fm(m, s):
    s = 0.0 if pd.isna(s) else s
    d = 1 if m < 100 else 0
    return f"{m:.{d}f}$\\pm${s:.{d}f}"

def agg(sub, kol):
    g = sub.groupby("model")[kol].agg(["mean", "std"])
    return g.reindex(MODELS)

def tulis(nama, teks):
    (OUT / f"{nama}.tex").write_text(teks, encoding="utf-8")

def tabel_faktor(kode, judul_nilai, caption, label):
    sub = df[df.bagian == kode]
    if sub.empty:
        tulis(f"tab_{kode}", f"\\begin{{table}}[t]\\caption{{{caption}}}\\label{{{label}}}\\centering\\begin{{tabular}}{{c}}(belum dijalankan)\\end{{tabular}}\\end{{table}}")
        return
    nilai = list(dict.fromkeys(sub["nilai"]))
    baris = []
    for v in nilai:
        s = sub[sub.nilai == v]
        for k, tag in [("val_MAE", "val"), ("test_MAE", "test")]:
            a = agg(s, k)
            best = a["mean"].idxmin()
            sel = [(f"\\textbf{{{fm(a.loc[m,'mean'],a.loc[m,'std'])}}}" if m == best else fm(a.loc[m, 'mean'], a.loc[m, 'std'])) for m in MODELS]
            kiri = f"\\multirow{{2}}{{*}}{{{v}}}" if tag == "val" else ""
            baris.append(f"{kiri} & {tag} & " + " & ".join(sel) + r" \\")
        baris.append(r"\addlinespace[1pt]")
    tulis(f"tab_{kode}", "\n".join([
        "\\begin{table}[t]", "\\centering", "\\scriptsize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}llcccc@{}}", "\\toprule",
        f"{judul_nilai} & MAE & ANN & RNN & LSTM & GRU \\\\", "\\midrule", *baris, "\\bottomrule", "\\end{tabular}", "\\end{table}"]))

NAMA = {"A1": ("$W$ (jam)", "Ablasi A1: panjang jendela $W$. MAE rerata$\\pm$sd (3 seed); tebal = model terbaik pada baris itu."),
        "A2": ("hidden", "Ablasi A2: ukuran hidden."),
        "A3": ("lapisan", "Ablasi A3: jumlah lapisan."),
        "A4": ("\\textit{lr}", "Ablasi A4: \\textit{learning rate}."),
        "A5": ("fitur", "Ablasi A5: kelompok fitur."),
        "B1": ("batch", "Ablasi B1: \\textit{batch size}."),
        "B2": ("dropout", "Ablasi B2: dropout.")}
for k, (jn, cap) in NAMA.items():
    tabel_faktor(k, jn, cap, f"tab:{k}")


# ---- Tabel gabungan A1-A5 + B (val & test MAE), untuk hemat ruang laporan
def tab_gabungan():
    baris = []
    for k, (jn, _) in NAMA.items():
        sub = df[df.bagian == k]
        if sub.empty: continue
        baris.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{k}: {jn}}}}} \\\\")
        for v in dict.fromkeys(sub["nilai"]):
            s_ = sub[sub.nilai == v]; sel = []
            for kol in ("val_MAE", "test_MAE"):
                a = agg(s_, kol); b = a["mean"].idxmin()
                sel += [(f"\\textbf{{{fm(a.loc[m,'mean'],a.loc[m,'std'])}}}" if m == b else fm(a.loc[m,'mean'],a.loc[m,'std'])) for m in MODELS]
            baris.append(f"\\hspace{{4pt}}{v.replace('_','')} & " + " & ".join(sel) + r" \\")
        baris.append("\\midrule")
    if not baris: return
    baris = baris[:-1]
    tulis("tab_AB", "\n".join(["\\begin{table*}[t]", "\\centering", "\\scriptsize", "\\caption{Ringkasan ablasi A1--A5 dan B (MAE rerata$\\pm$sd, 3 seed). Tebal = model terbaik pada baris dan split tersebut.}", "\\label{tab:AB}",
        "\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{@{}lcccccccc@{}}", "\\toprule",
        " & \\multicolumn{4}{c}{val MAE} & \\multicolumn{4}{c}{test MAE} \\\\", "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
        "Nilai & ANN & RNN & LSTM & GRU & ANN & RNN & LSTM & GRU \\\\", "\\midrule", *baris, "\\bottomrule", "\\end{tabular}", "\\end{table*}"]))
tab_gabungan()

# ---- Baseline naif + acuan (Bagian 0)
def naif_tab():
    f1, f2 = CSV.parent / "baseline_naif_duomo.csv", CSV.parent / "baseline_naif_lokasi.csv"
    if not (f1.exists() and f2.exists()):
        return
    a = pd.read_csv(f1).query("horizon == 1"); b = pd.read_csv(f2)
    b = b[b.horizon == 1] if "horizon" in b else b
    names = ["Naif: nilai terakhir", "Naif: kemarin", "Naif: minggu lalu"]
    baris = []
    for n in names:
        r = [n.replace("Naif: ", "")]
        for d in (a, b):
            for s in ("val", "test", "natal"):
                r.append(f"{d[(d.model == n) & (d.split == s)].MAE.item():.1f}")
        baris.append(" & ".join(r) + r" \\")
    tulis("tab_naif", "\n".join(["\\begin{table}[t]", "\\centering", "\\scriptsize", "\\caption{Baseline naif (MAE, horizon 1 jam).}", "\\label{tab:naif}",
        "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
        " & \\multicolumn{3}{c}{Duomo} & \\multicolumn{3}{c}{Lokasi kelompok} \\\\", "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "Tebakan & val & test & Natal & val & test & Natal \\\\", "\\midrule", *baris, "\\bottomrule", "\\end{tabular}", "\\end{table}"]))
naif_tab()

def tabel_banding(nama, bagian_map, caption, label):
    parts = []
    for ket, bag in bagian_map:
        s = df[df.bagian == bag]
        if not s.empty:
            parts.append((ket, s))
    if not parts:
        return
    baris = []
    for m in MODELS:
        for i, (ket, s) in enumerate(parts):
            x = s[s.model == m]
            if x.empty: continue
            c = [fm(x[k].mean(), x[k].std()) for k in ("val_MAE", "test_MAE", "natal_MAE")]
            c.append(f"{x['test_MAPE'].mean():.1f}")
            kiri = f"\\multirow{{{len(parts)}}}{{*}}{{{m}}}" if i == 0 else ""
            baris.append(f"{kiri} & {ket} & {int(x.param.iloc[0])} & " + " & ".join(c) + r" \\")
        baris.append(r"\addlinespace[1pt]")
    tulis(nama, "\n".join(["\\begin{table}[t]", "\\centering", "\\scriptsize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{2.5pt}", "\\begin{tabular}{@{}llrcccc@{}}", "\\toprule",
        "Model & Konf. & Param & val MAE & test MAE & Natal MAE & test MAPE (\\%) \\\\", "\\midrule", *baris, "\\bottomrule", "\\end{tabular}", "\\end{table}"]))

tabel_banding("tab_acuan", [("acuan", "0")], "Konfigurasi acuan di Duomo (rerata$\\pm$sd, 3 seed).", "tab:acuan_hasil")
cbag = [b for b in df.bagian.unique() if str(b).startswith("C")]
tabel_banding("tab_C", [("acuan", "0")] + [("terbaik" if b == "C" else b, b) for b in cbag], "Acuan vs.\\ konfigurasi terbaik (dipilih dengan val MAE) di Duomo.", "tab:C")
tabel_banding("tab_D", [("acuan", "D-acuan"), ("terbaik", "D-terbaik")], "Uji generalisasi di lokasi kelompok.", "tab:D")

# ---- Konfigurasi terbaik (teks)
def cfg_terbaik():
    s = df[df.bagian == "C"]
    if s.empty: return
    ac = {"W": 24, "lr": 0.001, "fitur": "semua", "batch": 64, "dropout": 0.0}
    hid = {"ANN": 32, "RNN": 68, "LSTM": 33, "GRU": 38}; lay = {"ANN": 2, "RNN": 1, "LSTM": 1, "GRU": 1}
    baris = []
    for m in MODELS:
        x = s[s.model == m]
        if x.empty: continue
        r = x.iloc[0]
        txt = f"$W{{=}}{int(r.W)}$, hidden {int(r.hidden)}, {int(r.layers)} lapisan, \\textit{{lr}} {r.lr:g}, fitur \\texttt{{{r.fitur.replace('_','')}}}, batch {int(r.batch)}, dropout {r.dropout:g}"
        baris.append(f"{m} & {txt} \\\\")
    tulis("tab_cfg", "\n".join(["\\begin{table}[t]", "\\centering", "\\scriptsize", "\\caption{Konfigurasi terbaik per model (Bagian C).}", "\\label{tab:cfg}",
        "\\begin{tabular}{@{}lp{6.6cm}@{}}", "\\toprule", "Model & Konfigurasi \\\\", "\\midrule", *baris, "\\bottomrule", "\\end{tabular}", "\\end{table}"]))
cfg_terbaik()

# ---- Sensitivitas: rentang val MAE rerata antar nilai faktor
PENDEK = {"A1": "jendela $W$", "A2": "hidden", "A3": "lapisan", "A4": "\\textit{lr}", "A5": "fitur", "B1": "batch", "B2": "dropout"}
sens, baris = {}, []
for k in NAMA:
    s = df[df.bagian == k]
    if s.empty: continue
    r = []
    for m in MODELS:
        g = s[s.model == m].groupby("nilai")["val_MAE"].mean()
        r.append(g.max() - g.min())
    sens[k] = r
    baris.append(f"{k} {PENDEK[k]} & " + " & ".join(f"{v:.0f}" for v in r) + r" \\")
if baris:
    tulis("tab_sens", "\n".join(["\\begin{table}[t]", "\\centering", "\\scriptsize", "\\caption{Sensitivitas: selisih val MAE rerata terbesar--terkecil antar nilai suatu faktor (makin besar = makin berpengaruh).}", "\\label{tab:sens}",
        "\\begin{tabular}{@{}lcccc@{}}", "\\toprule", "Faktor & ANN & RNN & LSTM & GRU \\\\", "\\midrule", *baris, "\\bottomrule", "\\end{tabular}", "\\end{table}"]))

# ---- Grafik gabungan
def panel(ax, k, split):
    s = df[df.bagian == k]
    if s.empty: return
    urut = list(dict.fromkeys(s["nilai"]))
    for m in MODELS:
        r = s[s.model == m].groupby("nilai")[f"{split}_MAE"].agg(["mean", "std"]).reindex(urut)
        x = np.arange(len(urut))
        ax.plot(x, r["mean"], "o-", ms=3, lw=1, color=WARNA[m], label=m)
        ax.fill_between(x, r["mean"] - r["std"].fillna(0), r["mean"] + r["std"].fillna(0), color=WARNA[m], alpha=.15)
    ax.set_xticks(np.arange(len(urut)), [u.replace("net+call+sms", "n+c+s").replace("net+waktu", "n+w").replace("semua", "all") for u in urut], fontsize=6)
    ax.tick_params(axis="y", labelsize=6); ax.grid(alpha=.25)
    ax.set_title(f"{k} ({split})", fontsize=7)

def grid(kodes, nama, lebar, tinggi):
    fig, axes = plt.subplots(2, len(kodes), figsize=(lebar, tinggi), squeeze=False)
    for j, k in enumerate(kodes):
        for i, sp in enumerate(["val", "test"]):
            panel(axes[i, j], k, sp)
    axes[0, 0].set_ylabel("MAE", fontsize=7); axes[1, 0].set_ylabel("MAE", fontsize=7)
    axes[0, -1].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(CSV.parent / nama, dpi=220); plt.close()
grid(["A1", "A2", "A3", "A4", "A5"], "fig_A.png", 7.2, 3.0)
grid(["B1", "B2"], "fig_B.png", 3.4, 3.0)

# ---- Makro & ringkasan
tot = df.drop_duplicates("kunci").detik.sum() / 60
(OUT / "makro.tex").write_text(f"\\newcommand{{\\totalmenit}}{{{tot:.0f}}}\n\\newcommand{{\\jumlahrun}}{{{len(df)}}}\n", encoding="utf-8")
with open(CSV.parent / "ringkasan.txt", "w") as f:
    f.write(f"{len(df)} baris, total training {tot:.1f} menit\n")
    f.write("Sensitivitas (rentang val MAE) [ANN RNN LSTM GRU]:\n")
    for k, r in sens.items(): f.write(f"  {k}: " + " ".join(f"{v:.0f}" for v in r) + "\n")
    cek = df.groupby(["bagian", "model", "nilai"]).seed.nunique()
    f.write(f"Konfigurasi dengan < 3 seed: {(cek < 3).sum()}\n")
print("selesai ->", OUT)
