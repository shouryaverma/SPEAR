import sys
sys.path.append("/depot/natallah/data/shourya/scDFM")
from src.data_process.data import Data

data = Data(data_path='/depot/natallah/data/shourya/scDFM/data')
data.load_data(data_name='norman')
data.process_data(n_top_genes=2000, infer_top_gene=1000, split_method='additive', use_negative_edge=True, k=30, fold=1)
genes = data.adata_test.var_names.tolist()
print("Extracted genes count:", len(genes))
print(genes[:10])
import pickle
with open("/depot/natallah/data/shourya/scDFM/bio_analysis/test_genes.pkl", "wb") as f:
    pickle.dump(genes, f)
