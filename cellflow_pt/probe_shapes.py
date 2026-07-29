import numpy as np
from src.data_process.data import Data, PerturbationDataset
from torch.utils.data import DataLoader
dm = Data('./data'); dm.load_data('norman')
dm.process_data(n_top_genes=5000, infer_top_gene=1000, split_method='additive', fold=1,
                use_negative_edge=True, k=30, split_file='', train_frac=1.0)
train_sampler, test_dl, _ = dm.load_flow_data(batch_size=64)
print("dm.adata", dm.adata.shape)
print("dm.adata_train", dm.adata_train.X.shape)
print("dm.adata_test", dm.adata_test.X.shape if hasattr(dm,'adata_test') else 'NA')
print("test_dl.adata", test_dl.adata.shape)
c = test_dl.get_control_data()
print("test control src", c['src_cell_data'].shape)
conds = list(test_dl._perturbation_covariates)
p = test_dl.get_perturbation_data(conds[0])
print("test pert tgt", p['tgt_cell_data'].shape, "cond_id", p['condition_id'].shape)
ds = PerturbationDataset(train_sampler, 64, cell_type_col='cell_type')
b = ds[0]
print("train batch src", np.array(b['src_cell_data']).shape, "tgt", np.array(b['tgt_cell_data']).shape,
      "cond", np.array(b['condition_id']).shape)
print("train_sampler.adata", train_sampler.adata.shape)
