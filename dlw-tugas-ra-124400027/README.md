# Studi Ablasi RNN, LSTM, GRU vs ANN untuk Prediksi Trafik Seluler Milan

Tugas Workshop Deep Learning — Teknik Telekomunikasi ITERA.

## Kelompok

| Peran | Nama | NIM |
|---|---|---|
| Ketua (mandiri) | Duta Rizky Pratama | 124400027 |

- **Kelas:** RA
- **Lokasi kelompok:** 7 — Università Bicocca (kotak 7467), digit terakhir NIM terkecil (124400027)
- **Laporan:** `laporan-latex/main.pdf`

## Cara Menjalankan

```bash
uv sync
uv run jupyter lab          # buka tugas_ablasi.ipynb, Run All (CEPAT = False, LOKASI_KELOMPOK = 7)
python buat_tabel.py        # membuat hasil/tabel/*.tex dan hasil/fig_A.png, fig_B.png dari hasil/ablasi.csv
cd laporan-latex && latexmk -pdf main.tex
```

Di Google Colab: pilih runtime GPU (T4), unggah folder ini, jalankan `!pip install torch pandas matplotlib` bila perlu, lalu Run All. Run yang terputus dilewati otomatis saat dijalankan ulang.

Perangkat yang dipakai: Google Colab (GPU NVIDIA T4), total waktu training ≈ 20 menit (lihat `hasil/ringkasan.txt`).

## Ringkasan Hasil

Pada acuan (Duomo), test MAE: RNN 230±5, GRU 237±9, LSTM 274±25, ANN 310±6 (baseline naif terbaik 468). Konfigurasi terbaik (dipilih dengan val MAE) memberi test MAE ANN 279±13, RNN 242±13, LSTM 247±13, GRU 230±8; val MAE gabungan hanya membaik untuk ANN. Di Università Bicocca selisih antar model menyusut ke level derau seed, dan pada minggu Natal semua model kalah dari tebakan naif ``nilai terakhir''. Detail di laporan dan `hasil/ablasi.csv` (348 run, 3 seed).

## Struktur

```
tugas_ablasi.ipynb      notebook eksperimen (di-commit beserta output)
buat_tabel.py           CSV -> tabel LaTeX (rerata ± sd) dan grafik laporan
data/                   data trafik per jam (Duomo + 10 lokasi tugas)
hasil/                  ablasi.csv (semua run), tabel, grafik
laporan-latex/          sumber laporan IEEE + PDF
```

## Data

Telecom Italia Big Data Challenge (Barlacchi dkk., 2015, *Scientific Data*, ODbL), versi per jam dari Zhang dkk. (INFOCOM 2021). Disiapkan oleh pengajar di repo [mctosima/dl-workshop](https://github.com/mctosima/dl-workshop).
