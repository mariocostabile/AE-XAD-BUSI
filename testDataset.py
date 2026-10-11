import json
from pathlib import Path

PROJECT = Path(r"D:\UNIVERSITÀ\Tesi\filenotebook2")
RAW_BENIGN    = PROJECT / "busi_raw" / "Dataset_BUSI_with_GT" / "benign"
RAW_MALIGNANT = PROJECT / "busi_raw" / "Dataset_BUSI_with_GT" / "malignant"

# --- 1. Trova i file immagine che hanno una maschera multipla (_mask_1, _mask_2, ...) ---
def find_multimask(folder):
    multi = set()
    for m in folder.glob("*_mask_*.png"):        # es. "benign (4)_mask_1.png"
        stem = m.name.split("_mask_")[0]          # -> "benign (4)"
        multi.add(stem + ".png")                  # -> "benign (4).png"
    return multi

multi_benign    = find_multimask(RAW_BENIGN)
multi_malignant = find_multimask(RAW_MALIGNANT)

print(f"Multi-maschera benign:    {len(multi_benign)}")
print(f"Multi-maschera malignant: {len(multi_malignant)}")
print("  benign:", sorted(multi_benign))
print("  malig :", sorted(multi_malignant))

all_multi = {("benign", f) for f in multi_benign} | {("malignant", f) for f in multi_malignant}

# --- 2. Per ogni seed e per ogni cartella HARDCODATA ---
SEEDS = [101, 237, 359, 461, 491]

# Inserisco a mano i percorsi completi visti nello screen (saltando la 256x14)
# Formato ipotizzato: cartella_esterna / cartella_interna_run
RUNS_FOLDERS = [
    PROJECT / "run_224_newLayer_folder_32x28" / "runs_224_newLayer_32x28",
    PROJECT / "run_224_newLayer_folder_64x14" / "runs_224_newLayer_64x14",
    PROJECT / "run_224_newLayer_folder_dense" / "runs_224_newLayer_dense",
    PROJECT / "run_224_noLayer_folder" / "runs_224_noLayer" 
]

for runs_base in RUNS_FOLDERS:
    print(f"\n{'='*70}")
    # Stampo il nome della cartella padre per chiarezza (es. run_224_newLayer_folder_dense)
    print(f"ANALIZZO CARTELLA: {runs_base.parent.name}")
    print(f"{'='*70}")
    
    for seed in SEEDS:
        taf = runs_base / f"seed{seed}" / "test_anom_files.json"
        
        if not taf.exists():
            print(f"  seed {seed}: json mancante in {taf} — salto")
            continue
            
        test_set = {(e["cls"], e["fname"]) for e in json.loads(taf.read_text())}
        in_test  = all_multi & test_set
        in_train = all_multi - test_set
        
        print(f"\nseed {seed}: {len(in_test)} in TEST (fix gratis)  |  {len(in_train)} in TRAIN (da valutare)")
        if in_train:
            print(f"   >>> in TRAIN: {sorted(in_train)}")