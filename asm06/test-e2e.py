# -*- coding: utf-8 -*-
"""Test end-to-end ASM06: notebooks, figures, models, meta, tài liệu — chạy thật trên đĩa.

Chạy: <venv> test-e2e.py  → in PASS/FAIL từng mục + exit code.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).parent
FIG = ROOT / "figures"
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


import nbformat  # noqa: E402

# 1) 5 notebook: tồn tại, 0 cell lỗi, có output
NBS = [
    "concepts/notebook/00_rnn_concepts.ipynb",
    "goldsilver/notebook/01-metals-rnn-pytorch.ipynb",
    "goldsilver/notebook/02-metals-rnn-keras.ipynb",
    "amzn/notebook/03-amzn-rnn-pytorch.ipynb",
    "amzn/notebook/04-amzn-rnn-keras.ipynb",
]
for rel in NBS:
    p = ROOT / rel
    if not p.exists():
        check(f"notebook {rel}", False, "thiếu file")
        continue
    nb = nbformat.read(p, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    errs = [c for c in code if any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    empty = [c for c in code if not c.get("outputs") and "def " not in c.source]
    check(f"notebook {rel}", not errs and len(code) >= 10,
          f"{len(nb.cells)} cells ({len(code)} code), {len(errs)} lỗi, {len(empty)} code cell không output")

# 2) figures đủ bộ
for pref, lo, hi in [("rnn", 1, 5), ("metals", 1, 15), ("amzn", 1, 9)]:
    have = sorted(FIG.glob(f"{pref}-*.png"))
    check(f"figures {pref}-*", len(have) >= hi, f"{len(have)} file")

# 3) model + meta
MODELS = [
    "goldsilver/model/metals_rnn_pytorch.pth",
    "goldsilver/model/metals_pytorch_meta.json",
    "goldsilver/model/metals_rnn_keras.keras",
    "goldsilver/model/metals_keras_meta.json",
    "amzn/model/amzn_rnn_pytorch.pth",
    "amzn/model/amzn_pytorch_meta.json",
    "amzn/model/amzn_rnn_keras.keras",
    "amzn/model/amzn_keras_meta.json",
    "concepts/model/rnn_concepts_sine.npz",
]
for rel in MODELS:
    p = ROOT / rel
    check(f"model {rel}", p.exists() and p.stat().st_size > 1000)

# 4) meta metrics hợp lý (RMSE dương, có naive để đối chiếu)
for rel in ["goldsilver/model/metals_pytorch_meta.json", "goldsilver/model/metals_keras_meta.json",
            "amzn/model/amzn_pytorch_meta.json", "amzn/model/amzn_keras_meta.json"]:
    m = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    met = m.get("metrics", {})
    rmse = met.get("rmse")
    naive = (met.get("naive") or {}).get("rmse")
    check(f"meta {rel}", isinstance(rmse, (int, float)) and rmse > 0 and naive is not None,
          f"rmse={rmse and round(rmse, 3)}, naive={naive and round(naive, 3)}, hidden={m.get('config', {}).get('hidden')}")

# 5) tài liệu bàn giao
DOCS = [
    "report/TAI_LIEU_KIEN_THUC_RNN_A06.docx",
    "report/TAI_LIEU_KIEN_THUC_RNN_A06.pdf",
    "report/CHUYEN_DE_RNN_A06.pptx",
    "report/KICH_BAN_THUYET_TRINH_A06.docx",
    "report/KICH_BAN_THUYET_TRINH_A06.pdf",
]
for rel in DOCS:
    p = ROOT / rel
    ok = p.exists() and p.stat().st_size > 30_000
    extra = ""
    if rel.endswith(".pptx"):
        from pptx import Presentation as _P
        n_sl = len(_P(str(p)).slides._sldIdLst)
        ok = ok and 1 <= n_sl <= 9
        extra = f", {n_sl} slides"
    check(f"doc {rel}", ok, (f"{p.stat().st_size // 1024} KB{extra}" if p.exists() else "thiếu"))

# 6) không chứa cụm bị cấm trong notebook + builder
BANNED = ["dành cho người mới học", "DÀNH CHO NGƯỜI MỚI"]
hits = []
for p in list(ROOT.glob("*/*/notebook/*.ipynb")) + [ROOT / "report"]:
    if p.is_dir():
        for f in p.glob("build-*.py"):
            t = f.read_text(encoding="utf-8", errors="ignore")
            hits += [f.name for b in BANNED if b.lower() in t.lower()]
    else:
        t = p.read_text(encoding="utf-8", errors="ignore")
        hits += [p.name for b in BANNED if b.lower() in t.lower()]
check("không cụm 'dành cho người mới học'", not hits, str(set(hits)))

print()
if FAILS:
    print(f"KẾT QUẢ: {len(FAILS)} FAIL →", FAILS)
    sys.exit(1)
print("KẾT QUẢ: TẤT CẢ PASS ✅")
