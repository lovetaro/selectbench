#!/usr/bin/env python3
"""CLI wrapper around CMD-GEN's GMM clustering step.

Upstream ``DiffPhar/get_phar/GMM_json.py`` has hardcoded input/output paths
and does matplotlib plotting. This keeps the same mapping and clustering
logic but takes ``--input`` / ``--output`` / ``--n_clusters`` as CLI args
and skips the plot.

Runs under the ``diff_phar`` conda env (needs sklearn).

Usage:
    python cmdgen_gmm.py --input phar.json --output phar.posp --n_clusters 7
"""

import argparse
import json

import numpy as np
from sklearn.mixture import GaussianMixture

# Derived from zyrlia1018/CMD-GEN @ 8ab69ea (DiffPhar/get_phar/GMM_json.py),
# MIT licence, Copyright (c) 2024 zyrlia1018. See LICENSE.upstream.
# Copied verbatim from DiffPhar/get_phar/GMM_json.py so GCPG's expected
# feature-type vocabulary stays in sync.
MAPPING = {
    "Aromatic": 1,
    "Hydrophobe": 2,
    "PosIonizable": 3,
    "Acceptor": 4,
    "Donor": 5,
    "LumpedHydrophobe": 6,
    "others": 7,
}
IDX2PHAR = {1: "AROM", 2: "HYBL", 3: "POSC", 4: "HACC",
            5: "HDON", 6: "LHYBL", 7: "UNKNOWN"}


def _filter_outliers(X, k=5.0):
    """Drop points whose per-axis robust z-score (vs median, scaled by MAD)
    exceeds k. Prevents stray DiffPhar samples from anchoring a GMM cluster
    far outside the real binding site (observed for parp1: a single z=55.59
    point ~25 Å above the pocket dragged GCPG into all-alkyne molecules)."""
    median = np.median(X, axis=0)
    mad = np.median(np.abs(X - median), axis=0)
    sigma = np.where(mad > 0, mad * 1.4826, 1.0)  # MAD -> sigma-equivalent
    z = np.abs(X - median) / sigma
    keep = (z <= k).all(axis=1)
    return X[keep], int((~keep).sum())


def run_gmm(input_json, output_posp, n_clusters=7, seed=42):
    with open(input_json) as f:
        phar = json.load(f)

    vectors = []
    for _molecule, features in phar.items():
        for _feature, coords in features.items():
            vectors.extend(coords)
    X = np.array(vectors)

    X_fit, n_dropped = _filter_outliers(X, k=5.0)
    if n_dropped > 0:
        print(f"[filter] dropped {n_dropped}/{len(X)} outlier points "
              f"(>5·MAD from per-axis median) before GMM fit")

    gmm = GaussianMixture(n_components=n_clusters, random_state=seed)
    gmm.fit(X_fit)
    centers = gmm.means_

    feature_probs = {feat: np.zeros(n_clusters)
                     for mol_features in phar.values() for feat in mol_features}
    for _mol, features in phar.items():
        for feat, coords in features.items():
            probs = np.sum(gmm.predict_proba(coords), axis=0)
            feature_probs[feat] += probs

    for feat in feature_probs:
        s = np.sum(feature_probs[feat])
        if s > 0:
            feature_probs[feat] /= s

    lines = []
    for i in range(n_clusters):
        ranked = sorted(feature_probs.keys(),
                        key=lambda f, i=i: feature_probs[f][i], reverse=True)
        most_common = ranked[0]
        if most_common not in MAPPING:
            continue
        phar_type = IDX2PHAR[MAPPING[most_common]]
        cx, cy, cz = centers[i]
        lines.append(f"{phar_type} {cx:.2f} {cy:.2f} {cz:.2f}")

    with open(output_posp, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(lines)} pharmacophore points to {output_posp}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="input .json from generate_phars.py")
    p.add_argument("--output", required=True, help="output .posp path")
    p.add_argument("--n_clusters", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_gmm(args.input, args.output, args.n_clusters, args.seed)


if __name__ == "__main__":
    main()
