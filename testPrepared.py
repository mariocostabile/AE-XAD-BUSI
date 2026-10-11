# verifica_busi_prepared.py
# Controlla che busi_prepared abbia le maschere FUSE (non parziali).
# Confronta, per ogni immagine multi-maschera, l'area in busi_prepared
# contro l'area attesa fondendo le maschere RAW.

import numpy as np
from PIL import Image
from pathlib import Path

PROJECT = Path(r"D:\UNIVERSITÀ\Tesi\filenotebook2")
RAW     = PROJECT / "busi_raw" / "Dataset_BUSI_with_GT"
PREP    = PROJECT / "busi_prepared" / "test"
RESIZE  = (224, 224)

def find_multimask(folder):
    """immagini che hanno almeno un _mask_1, _mask_2, ..."""
    multi = set()
    for m in folder.glob("*_mask_*.png"):
        multi.add(m.name.split("_mask_")[0] + ".png")
    return sorted(multi)

def area_raw_fusa(cls, fname):
    """area (n pixel >0) fondendo TUTTE le maschere raw, scalata a 224 NEAREST."""
    stem = Path(fname).stem
    folder = RAW / cls
    mask_files = [folder / f"{stem}_mask.png"] + sorted(folder.glob(f"{stem}_mask_*.png"))
    mask_files = [m for m in mask_files if m.exists()]
    merged = None
    for mp in mask_files:
        m = (np.array(Image.open(mp).convert("L")) > 0).astype(np.uint8)
        merged = m if merged is None else (merged | m)
    m255 = (merged * 255).astype(np.uint8)
    m_res = np.array(Image.fromarray(m255).resize(RESIZE, Image.NEAREST)) > 0
    return int(m_res.sum())

def area_prepared(cls, fname):
    """area della maschera in busi_prepared (già 224)."""
    p = PREP / cls / "masks" / fname
    if not p.exists():
        return None
    m = np.array(Image.open(p).convert("L")) > 0
    return int(m.sum())

print("="*70)
print("VERIFICA busi_prepared — maschere fuse?")
print("="*70)

problemi = 0
totali   = 0
for cls in ("benign", "malignant"):
    multi = find_multimask(RAW / cls)
    print(f"\n[{cls}] {len(multi)} immagini multi-maschera")
    for fname in multi:
        totali += 1
        att = area_raw_fusa(cls, fname)      # area attesa (fusa)
        got = area_prepared(cls, fname)      # area in busi_prepared
        if got is None:
            print(f"  ✗ {fname:24s} ASSENTE in busi_prepared")
            problemi += 1
            continue
        # tolleranza: il resize NEAREST puo' variare di qualche px.
        # Il segnale forte e' se got e' MOLTO minore di att (= ancora parziale).
        ok = got >= 0.85 * att
        flag = "OK " if ok else "✗ PARZIALE"
        if not ok:
            problemi += 1
        print(f"  {flag} {fname:24s} prepared={got:5d}  atteso(fuso)={att:5d}")

print("\n" + "="*70)
if problemi == 0:
    print(f"✓ TUTTE le {totali} maschere multi sono FUSE correttamente in busi_prepared.")
else:
    print(f"✗ {problemi}/{totali} maschere NON fuse (ancora parziali o assenti).")
    print("  -> busi_prepared NON e' a posto: rigenera (svuota test/*/masks + cella 10).")
print("="*70)