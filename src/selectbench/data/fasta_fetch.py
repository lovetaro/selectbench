import re

import requests

FASTA_URL = "https://www.rcsb.org/fasta/entry/"


def _fetch_single_chain_sequence(fasta_text, pdb_id, chain_id):
    """Parse FASTA text and return the sequence for a single chain_id, or None."""

    def _chain_matches(header, chain_id):
        # Header format: >PDBID_N|Chains A[auth A]|description|organism
        # Label and auth chains may differ when RCSB renames chains.
        # Handles formats:
        #   "Chains A"
        #   "Chains A[auth B]"
        #   "Chains A,B[auth C,D]"
        #   "Chains A[auth M], F[auth R]"   ← per-chain auth annotations
        parts = header.split("|")
        if len(parts) < 2:
            return False
        chains_field = parts[1]

        chains_str = re.sub(r"^\s*Chains?\s*", "", chains_field)

        label_chains = []
        auth_chains = []

        # Each "X[auth Y]" or "X[auth Y,Z]" block
        for m in re.finditer(r"([A-Za-z0-9]+)\s*\[auth\s+([^\]]+)\]", chains_str):
            label_chains.append(m.group(1).strip())
            for a in m.group(2).split(","):
                a = a.strip()
                if a:
                    auth_chains.append(a)

        # Standalone labels (not attached to an [auth ...] block)
        remaining = re.sub(r"[A-Za-z0-9]+\s*\[auth\s+[^\]]+\]", "", chains_str)
        for tok in re.split(r"[\s,]+", remaining):
            tok = tok.strip()
            if tok:
                label_chains.append(tok)

        return chain_id in label_chains or chain_id in auth_chains

    current_header = None
    current_seq = []
    for line in fasta_text.splitlines():
        if line.startswith(">"):
            if current_header is not None and _chain_matches(current_header, chain_id):
                return "".join(current_seq)
            current_header = line
            current_seq = []
        else:
            current_seq.append(line.strip())

    if current_header is not None and _chain_matches(current_header, chain_id):
        return "".join(current_seq)

    print(f"Chain {chain_id} not found in FASTA for {pdb_id}")
    return None


def fetch_protein_sequence(pdb_id, chain_id):
    """Fetch the protein sequence(s) for a chain ID from RCSB PDB FASTA.

    chain_id may be a single chain (e.g. "A") or a bracketed semicolon-separated
    list of chains (e.g. "[A;B;C;D]"), in which case the result is returned as
    "[seq1;seq2;seq3;seq4]".
    """
    r = requests.get(FASTA_URL + pdb_id.upper(), timeout=30)
    if r.status_code != 200:
        print(f"Failed to fetch FASTA for {pdb_id} (HTTP {r.status_code})")
        return None

    fasta_text = r.text

    # Detect multi-chain format: [A;B;C;D]
    if chain_id.startswith("[") and chain_id.endswith("]"):
        chains = [c.strip() for c in chain_id[1:-1].split(";")]
        seqs = []
        for c in chains:
            seq = _fetch_single_chain_sequence(fasta_text, pdb_id, c)
            seqs.append(seq if seq is not None else "")
        return "[" + ";".join(seqs) + "]"

    return _fetch_single_chain_sequence(fasta_text, pdb_id, chain_id)