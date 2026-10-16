# build_grn_edges.py
# PYTHONPATH=./ python src/data_process/build_grn_edges.py
import gzip
import pandas as pd
from src.data_process.data import Data
from src.utils.utils import process_vocab

DATA_NAME = "norman"
N_TOP_GENES = 5000
INFER_TOP_GENE = 1000
SPLIT_METHOD = "additive"
FOLD = 1
TOPK = 30
USE_NEGATIVE_EDGE = True

STRING_LINKS = "/depot/natallah/data/shourya/scDFM/data/string/9606.protein.links.v12.0.txt"
STRING_INFO = "/depot/natallah/data/shourya/scDFM/data/string/9606.protein.info.v12.0.txt"
OUT_CSV = "data/norman/grn_edges.csv"
SCORE_CUTOFF = 700   # STRING combined_score is 0-1000; 700 = high confidence


class _FakeConfig:
    data_name = DATA_NAME
    n_top_genes = N_TOP_GENES


def open_maybe_gzip(path):
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt")
    return open(path, "rt")


# 1. Exact HVG symbols, same pipeline as run.py.
data_manager = Data("./data")
data_manager.load_data(DATA_NAME)
data_manager.process_data(
    n_top_genes=N_TOP_GENES, infer_top_gene=INFER_TOP_GENE,
    split_method=SPLIT_METHOD, fold=FOLD,
    use_negative_edge=USE_NEGATIVE_EDGE, k=TOPK,
)
process_vocab(data_manager, _FakeConfig())
hvg = set(str(n) for n in data_manager.adata.var_names)
print(f"HVG genes: {len(hvg)}")

# 2. STRING protein id -> gene symbol (info file is tab-delimited).
id_to_symbol = {}
with open_maybe_gzip(STRING_INFO) as f:
    header = f.readline().lstrip("#").rstrip("\n").split("\t")
    try:
        id_col = header.index("string_protein_id")
        name_col = header.index("preferred_name")
    except ValueError:
        id_col, name_col = 0, 1   # fallback to positional
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) <= max(id_col, name_col):
            continue
        id_to_symbol[parts[id_col]] = parts[name_col]
print(f"STRING proteins mapped: {len(id_to_symbol)}")

# 3. Filter edges. Links file is space-delimited; locate score column by header.
with open_maybe_gzip(STRING_LINKS) as f:
    link_header = f.readline().rstrip("\n").split()
    if "combined_score" not in link_header:
        raise RuntimeError(
            f"{STRING_LINKS} is not a STRING links file "
            f"(no 'combined_score' column; header={link_header[:6]}). "
            f"Re-download 9606.protein.links.v12.0.txt.gz."
        )
    p1_col = link_header.index("protein1")
    p2_col = link_header.index("protein2")
    score_col = link_header.index("combined_score")
    max_col = max(p1_col, p2_col, score_col)

    seen = set()
    rows = []
    for line in f:
        parts = line.split()
        if len(parts) <= max_col:
            continue
        try:
            score = int(parts[score_col])
        except ValueError:
            continue
        if score < SCORE_CUTOFF:
            continue
        a = id_to_symbol.get(parts[p1_col])
        b = id_to_symbol.get(parts[p2_col])
        if a is None or b is None or a == b:
            continue
        if a not in hvg or b not in hvg:
            continue
        key = (a, b) if a < b else (b, a)   # STRING lists both directions; dedup
        if key in seen:
            continue
        seen.add(key)
        rows.append(key)

pd.DataFrame(rows, columns=["gene_a", "gene_b"]).to_csv(OUT_CSV, index=False)
print(f"Wrote {len(rows)} edges to {OUT_CSV}")