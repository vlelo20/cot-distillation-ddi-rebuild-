"""
Step 9: Generate Morgan fingerprints from SMILES.

For each drug with a valid SMILES string, compute a 2048-bit Morgan fingerprint
(radius=2, equivalent to ECFP4). These are the standard molecular fingerprints
for drug similarity work.

Coverage: only drugs with non-empty SMILES (small molecules, ~14,627 of 19,857).
Biotech drugs without SMILES get no fingerprint -- handled in Step 10/11 with
fallback strategies.

Inputs:
  processed_v2/drug_profiles.json

Outputs:
  processed_v2/drug_fingerprints.pkl  -- {drugbank_id: numpy_bit_array}
  reports/09_fingerprints.txt
"""
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit import DataStructs

# Suppress rdkit's noisy warnings about kekulization, valence, etc.
# These fire on weird drugs (charged species, exotic stereochemistry) and
# don't affect fingerprinting -- the molecule still gets a fingerprint.
RDLogger.DisableLog("rdApp.*")

ROOT = Path.home() / "ddiproject"
PROCESSED = ROOT / "processed_v2"
REPORT = ROOT / "reports" / "09_fingerprints.txt"

FP_RADIUS = 2          # ECFP4 equivalent
FP_BITS = 2048         # standard size; 1024 is also common, 2048 gives lower collision rate


def smiles_to_fingerprint(smiles, radius=FP_RADIUS, n_bits=FP_BITS):
    """
    Convert a SMILES string to a Morgan fingerprint as a numpy uint8 bit array.
    Returns None if SMILES can't be parsed (rare but happens with malformed entries).
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # GetMorganFingerprintAsBitVect returns an ExplicitBitVect; convert to numpy
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def main():
    print("Loading drug profiles...")
    with open(PROCESSED / "drug_profiles.json") as f:
        profiles = json.load(f)
    print(f"  {len(profiles):,} drugs total")

    fingerprints = {}
    n_no_smiles = 0
    n_parse_failed = 0
    n_success = 0
    failed_examples = []

    print(f"\nGenerating Morgan fingerprints (radius={FP_RADIUS}, bits={FP_BITS})...")
    for did in tqdm(profiles, desc="fingerprinting", unit="drug"):
        smiles = profiles[did].get("smiles", "")
        if not smiles:
            n_no_smiles += 1
            continue
        fp = smiles_to_fingerprint(smiles)
        if fp is None:
            n_parse_failed += 1
            if len(failed_examples) < 5:
                failed_examples.append((did, profiles[did]["name"], smiles[:80]))
            continue
        fingerprints[did] = fp
        n_success += 1

    print(f"\nResults:")
    print(f"  Successful:        {n_success:,}")
    print(f"  No SMILES:         {n_no_smiles:,}  (biotech drugs)")
    print(f"  SMILES parse fail: {n_parse_failed:,}")

    print(f"\nWriting drug_fingerprints.pkl...")
    with open(PROCESSED / "drug_fingerprints.pkl", "wb") as f:
        pickle.dump(fingerprints, f, protocol=pickle.HIGHEST_PROTOCOL)

    out_size_mb = (PROCESSED / "drug_fingerprints.pkl").stat().st_size / 1e6
    print(f"  size: {out_size_mb:.1f} MB")

    # Quick sanity check: pick one drug and show fingerprint stats
    if "DB00945" in fingerprints:  # Aspirin -- well-known small molecule
        fp = fingerprints["DB00945"]
        print(f"\nSanity check (DB00945 = Aspirin):")
        print(f"  fingerprint shape: {fp.shape}, dtype: {fp.dtype}")
        print(f"  bits set: {int(fp.sum())} / {FP_BITS}")

    # Report
    coverage_total = 100 * n_success / len(profiles)
    coverage_smiles = 100 * n_success / max(n_success + n_parse_failed, 1)
    lines = [
        "Step 9 -- Morgan fingerprints",
        "=" * 60,
        f"Total drugs:                 {len(profiles):,}",
        f"With SMILES:                 {n_success + n_parse_failed:,}",
        f"Fingerprints generated:      {n_success:,}",
        f"  coverage of all drugs:     {coverage_total:.1f}%",
        f"  coverage of SMILES drugs:  {coverage_smiles:.2f}%",
        "",
        f"No SMILES (biotech):         {n_no_smiles:,}",
        f"SMILES parse failures:       {n_parse_failed:,}",
        "",
        f"Parameters:",
        f"  radius:    {FP_RADIUS}  (ECFP4 equivalent)",
        f"  bits:      {FP_BITS}",
    ]
    if failed_examples:
        lines.append("")
        lines.append("Sample SMILES parse failures:")
        for did, name, smiles in failed_examples:
            lines.append(f"  {did} {name}: {smiles}")
    lines.extend([
        "",
        "Outputs:",
        f"  drug_fingerprints.pkl  ({out_size_mb:.1f} MB)",
    ])
    report = "\n".join(lines)
    print()
    print(report)
    REPORT.write_text(report)


if __name__ == "__main__":
    main()
