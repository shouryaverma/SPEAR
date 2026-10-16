import scanpy as sc
import pandas as pd
import numpy as np
import json
import torch
import pickle
import re
from typing import Union, Optional
from pathlib import Path
import os
from src.utils._preprocessing import annotate_compounds, get_molecular_fingerprints
from src.data_process._datamanager import DataManager
import jax
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pdb
import tqdm
from random import shuffle
from src.utils.utils import build_gene_coexpression_graph,sorted_pad_mask
# combosciplex url: https://figshare.com/articles/dataset/combosciplex/25062230?file=44229635
# 'norman' url = 'https://dataverse.harvard.edu/api/access/datafile/6154020'

class Data:
    def __init__(self, data_path='../../data'):
        self.data_path = data_path
        if not os.path.exists(data_path):
            raise ValueError(data_path + ' does not exist')
            # os.makedirs(data_path)

        
    def load_data(self, data_name = None, data_path = None):
        self.data_name = data_name
        if data_name in ['norman', 'norman_umi_go_filtered', 'parse_pbmc', 'op3']:
            self.adata = sc.read_h5ad(os.path.join(self.data_path, data_name + '.h5ad'))
        elif data_name in ['combosciplex', ]:
            self.adata = sc.read_h5ad(os.path.join(self.data_path, data_name + '.h5ad'))
        elif data_name in ['replogle', ]:
            self.adata = sc.read_h5ad(os.path.join(self.data_path, data_name + '.h5ad'))
        elif data_name in ['dixit', ]:
            self.adata = sc.read_h5ad(os.path.join(self.data_path, data_name + '.h5ad'))
        else:
            raise ValueError(data_name + ' is not a valid data name')
        
    def process_data(self, n_top_genes = 2000,infer_top_gene=1000,split_method='additive',
                     use_negative_edge=True, k=30,
                     **kwargs):
        os.makedirs(os.path.join(self.data_path, self.data_name), exist_ok=True)
        if self.data_name == 'combosciplex':
            
            if os.path.exists(os.path.join(self.data_path, self.data_name, 'processed.h5ad')):
                self.adata = sc.read_h5ad(os.path.join(self.data_path, self.data_name, 'processed.h5ad'))
            else:   

                self.adata.obs["condition"] = self.adata.obs.apply(
                    lambda x: "control" if x["condition"] == "control+control" else x["condition"], axis=1
                )

                self.adata.obs["is_control"] = self.adata.obs.apply(
                    lambda x: True if x["condition"] == "control" else False, axis=1
                )
                
                try:
                    annotate_compounds(self.adata, compound_keys=["Drug1", "Drug2"])
                    get_molecular_fingerprints(self.adata, compound_keys=["Drug1", "Drug2"])
                    self.adata.uns["fingerprints"]["control"] = np.zeros(1024)
                except Exception as e:
                    # No network / pertpy: fall back to categorical drug tokens (no molecular
                    # fingerprints). Downstream crisper/embedder uses perturbation_dict, not uns.
                    print(f"[combosciplex] molecular fingerprints unavailable ({type(e).__name__}); "
                          f"using categorical drug tokens only")
                    self.adata.uns["fingerprints"] = {}

                self.adata.write(os.path.join(self.data_path, self.data_name, 'processed.h5ad'))
            
            self.adata.X = self.adata.layers["counts"].copy()
            sc.pp.normalize_total(self.adata)
            sc.pp.log1p(self.adata)
            sc.pp.highly_variable_genes(self.adata, inplace=True, n_top_genes=n_top_genes)
                
            if 'test_conditions' in kwargs.keys():
                test_conditions = kwargs['test_conditions']
            else:
                test_conditions = ['Panobinostat+Crizotinib', 
                                'Panobinostat+Curcumin', 
                                'Panobinostat+SRT1720', 
                                'Panobinostat+Sorafenib', 
                                'SRT2104+Alvespimycin', 
                                'control+Alvespimycin', 
                                'control+Dacinostat']
                
            self.adata = self.adata[:,self.adata.var['highly_variable']] # filter out low variable genes
            
            self.adata.obs["mode"] = self.adata.obs.apply(lambda x: "test" if x["condition"] in test_conditions else "train", axis=1)
            self.adata_train = self.adata[self.adata.obs["mode"] == "train"]
            self.adata_test = self.adata[(self.adata.obs["mode"] == "test") | (self.adata.obs["condition"]=="control")]
            
            sc.pp.highly_variable_genes(self.adata_test, inplace=True, n_top_genes=infer_top_gene)
            self.adata_test = self.adata_test[:,self.adata_test.var['highly_variable']]
            
            condition = np.unique(list(self.adata.obs['condition']))
            unique_perturbation = []
            np.array([unique_perturbation.extend(perturbation.split('+')) for perturbation in condition])
            unique_perturbation = np.unique(unique_perturbation)
            unique_perturbation.sort()
            self.unique_perturbation = unique_perturbation
            self.perturbation_dict = {perturbation: i for i, perturbation in enumerate(unique_perturbation)}
            # self._val_manager = 
        elif self.data_name in ['norman', 'norman_umi_go_filtered', 'parse_pbmc', 'op3', 'replogle', 'dixit']:

            if self.data_name == 'dixit':
                # Dixit K562 TF Perturb-seq (High MOI), raw counts. Guide names are
                # 'p-sgTF-N' (and '_'-joined for multi-guide cells). Parse each cell's
                # guide set -> TF set (intergenic/control guides yield nothing), then:
                #   0 TF -> 'control' (true control + intergenic-only cells),
                #   1 TF -> 'TF+control', 2 TF -> sorted 'A+B'.
                # Drop >=3-TF cells: SPEAR predicts doubles from singles (additive split),
                # and Drug1/Drug2 only hold two tokens. This keeps the 45 = C(10,2) fully
                # covered double combos + 10 singles + control -- all perturbations are TFs,
                # so every epistatic pair lands on a TF-TF regulatory edge (the control
                # variable norman lacks).
                p = self.adata.obs['perturbation'].astype(str)
                def _to_tfs(x):
                    if x == 'control':
                        return []
                    s = set()
                    for g in x.split('_'):
                        m = re.match(r'p-sg([A-Za-z0-9]+)-\d+', g)
                        if m:
                            s.add(m.group(1))
                    return sorted(s)
                tf_lists = p.map(_to_tfs)
                ntf = tf_lists.map(len)
                keep = (ntf <= 2).values
                self.adata = self.adata[keep].copy()
                tf_lists = tf_lists[keep]
                def _to_cond(tfs):
                    if len(tfs) == 0:
                        return 'control'
                    if len(tfs) == 1:
                        return f'{tfs[0]}+control'
                    return f'{tfs[0]}+{tfs[1]}'
                cond = tf_lists.map(_to_cond)
                self.adata.obs['condition'] = cond.values
                ctrl_mask = (self.adata.obs['condition'] == 'control').values
                sc.pp.normalize_total(self.adata)
                sc.pp.log1p(self.adata)
                self.adata.obs['control'] = ctrl_mask.astype(int)

            if self.data_name == 'replogle':
                # Single-gene Perturb-seq, raw counts. Build 'condition' from 'perturbation',
                # normalize+log, convert bare 'GENE' -> 'GENE+control' (SPEAR needs Drug1+Drug2),
                # control stays 'control', and build the 0/1 control column downstream needs.
                self.adata.obs['condition'] = self.adata.obs['perturbation'].astype(str)
                ctrl_mask = (self.adata.obs['condition'] == 'control').values
                sc.pp.normalize_total(self.adata)
                sc.pp.log1p(self.adata)
                self.adata.obs['condition'] = self.adata.obs['condition'].astype(str).apply(
                    lambda g: 'control' if g == 'control' else f'{g}+control')
                self.adata.obs['control'] = ctrl_mask.astype(int)

            sc.pp.highly_variable_genes(self.adata, inplace=True, n_top_genes=n_top_genes)
            
            self.adata.obs['condition'] = self.adata.obs['condition'].str.replace('ctrl', 'control', regex=False)

            unique_perturbation = []
            [unique_perturbation.extend(perturbation.split('+')) for perturbation in self.adata.obs['condition'].unique()]
            unique_perturbation = np.unique(unique_perturbation)
            
            if self.data_name in ['norman', 'replogle', 'dixit']:
                for perturbation in unique_perturbation:
                    if perturbation in self.adata.var_names:
                        self.adata.var.loc[perturbation, 'highly_variable'] = True
                    else:
                        print(f"Warning: {perturbation} is not in the gene names")
                self.adata = self.adata[:,self.adata.var['highly_variable']]
            elif self.data_name == 'norman_umi_go_filtered':
                all_gene_names = list(self.adata.var['gene_name']) + ['ctrl']
                for perturbation in unique_perturbation:
                    if perturbation not in all_gene_names:
                        print(f"Warning: {perturbation} is not in the gene names")
                self.adata.var['highly_variable'] = True
            else:
                self.adata = self.adata[:, self.adata.var['highly_variable']]

            
            #### for split five times
            if self.data_name == 'replogle':
                # Unseen single-gene holdout: hold out 20% of perturbed genes as test.
                # Model never trains on them -> must generalize via causal/GRN propagation
                # from seen genes. This is where causal structure is irreplaceable.
                singles = sorted([g for g in unique_perturbation if g != 'control'])
                self.split_results = []
                for i in range(5):
                    rng = np.random.RandomState(42 + i)
                    sh = np.array(singles).copy(); rng.shuffle(sh)
                    n_test = max(1, int(len(sh) * 0.2))
                    held = sh[:n_test]
                    self.split_results.append({'test': [f'{g}+control' for g in held]})
                print(f'[replogle] unseen split: {len(self.split_results[0]["test"])} held-out genes/fold '
                      f'of {len(singles)} total')
            elif split_method in ['additive', 'combinations']:
                split_file = os.path.join(self.data_path, self.data_name, 'split_results.pkl')
                if os.path.exists(split_file):
                    with open(split_file, 'rb') as f:
                        self.split_results = pickle.load(f)
                else:
                    perturbations = np.unique(self.adata.obs['condition'])
                    double_perturbation = [p for p in perturbations if 'ctrl' not in p]
                    double_perturbation = np.array(double_perturbation)

                    self.split_results = []
                    
                    for i in range(5):
                        np.random.seed(42 + i)
                        shuffled = double_perturbation.copy()
                        np.random.shuffle(shuffled)
                        
                        split_idx = int(len(shuffled) * 0.3)
                        test_double = shuffled[:split_idx]
                        train_double = shuffled[split_idx:]
                        self.split_results.append({
                            'train': train_double.tolist(),
                            'test': test_double.tolist()
                        })
                    
                    with open(split_file, 'wb') as f:
                        pickle.dump(self.split_results, f)
                    print('split results saved')
                    
            elif split_method == 'unseen':
                split_file = kwargs.get('split_file', '') or os.path.join(self.data_path, self.data_name, 'split_results_unseen.pkl')
                if os.path.exists(split_file):
                    with open(split_file, 'rb') as f:
                        self.split_results = pickle.load(f)
                else:
                    perturbations = np.unique(self.adata.obs['condition'])
                    # conditions are already normalized: 'control', 'GENE+control', 'A+B'
                    double_perturbation = [
                        p for p in perturbations
                        if p != 'control' and 'control' not in p.split('+')
                    ]
                    single_conditions = set(p for p in perturbations if p.endswith('+control'))
                    genes = sorted(set(
                        g for p in double_perturbation for g in p.split('+')
                    ))
                    if len(genes) == 0:
                        raise ValueError(f'{self.data_name} has no double perturbations; '
                                         f'split_method=unseen is undefined')
                    n_hold = min(12, max(1, len(genes) // 4))

                    self.split_results = []
                    for i in range(5):
                        rng = np.random.RandomState(42 + i)
                        shuffled = np.array(genes, dtype=object).copy()
                        rng.shuffle(shuffled)
                        remove_genes = set(shuffled[:n_hold].tolist())

                        p_count = {}
                        for p in double_perturbation:
                            ps = p.split('+')
                            p_count[p] = int(ps[0] in remove_genes) + int(ps[1] in remove_genes)
                        test = [p for p, c in p_count.items() if c > 0]
                        # hold out the singles of the removed genes too, otherwise the gene
                        # is seen in training and the split is not unseen
                        test.extend([f'{g}+control' for g in sorted(remove_genes)
                                     if f'{g}+control' in single_conditions])

                        self.split_results.append({
                            'remove_genes': sorted(remove_genes),
                            'p_count': p_count,
                            'test': test,
                        })
                    with open(split_file, 'wb') as f:
                        pickle.dump(self.split_results, f)
                    print(f'split results unseen saved: {n_hold} held-out genes, '
                          f'{len(self.split_results[0]["test"])} test conditions/fold')
            
            elif split_method == 'state_pbmc':
                DONOR_COL = 'donor'
                N_HOLDOUT_DONORS = 4

                split_file = os.path.join(self.data_path, self.data_name, 'split_results_state_pbmc.pkl')
                if os.path.exists(split_file):
                    with open(split_file, 'rb') as f:
                        self.split_results = pickle.load(f)
                else:
                    obs = self.adata.obs
                    all_donors = np.array(obs[DONOR_COL].dropna().astype(str).unique())
                    perturbations = [p for p in obs['condition'].unique() if p != 'control']
                    donors_of_pert = {
                        p: set(obs.loc[obs['condition'] == p, DONOR_COL].astype(str).unique())
                        for p in perturbations
                    }
                    self.split_results = []
                    for i in range(5):
                        rng = np.random.default_rng(42 + i)
                        held_out_donors = rng.permutation(all_donors)[:N_HOLDOUT_DONORS].tolist()
                        held_set = set(held_out_donors)
                        donor_test_perturbations = {}
                        for donor in held_out_donors:
                            donor_test_perturbations[donor] = sorted(
                                p for p in perturbations
                                if donor in donors_of_pert[p]
                                and len(donors_of_pert[p] - held_set) > 0
                            )
                        self.split_results.append({
                            'held_out_donors': held_out_donors,
                            'donor_test_perturbations': donor_test_perturbations,
                        })
                    with open(split_file, 'wb') as f:
                        pickle.dump(self.split_results, f)
                    print('split results state_pbmc saved')

            else:
                raise ValueError('split_method must be one of additive, combinations, unseen or state_pbmc')


            if 'fold' in kwargs.keys():
                fold = kwargs['fold']
            else:
                fold = 0

            self.adata.obs['Drug1'] = self.adata.obs['condition'].str.split('+').apply(lambda x: x[0])
            self.adata.obs['Drug2'] = self.adata.obs['condition'].str.split('+').apply(lambda x: x[-1])
            self.adata.obs['is_control'] = False
            self.adata.obs.loc[self.adata.obs['control'] == 1, 'is_control'] = True
            self.adata.obs['mode'] = 'train'
            
            
            if split_method == 'combinations':
                self.split_results[fold]['test'] = self.split_results[fold]['test'][:15]
                remove_genes = []
                [remove_genes.extend(p.split('+')) for p in self.split_results[fold]['test']]
                remove_genes = set(remove_genes)
                remove_genes_condition = [p+'+control' for p in remove_genes]
                
                self.split_results[fold]['test'].extend(remove_genes_condition)            
            
            
            if split_method == 'state_pbmc':
                DONOR_COL = 'donor'
                for donor in self.split_results[fold]['held_out_donors']:
                    test_perts = self.split_results[fold]['donor_test_perturbations'][donor]
                    mask = (self.adata.obs[DONOR_COL] == donor) & (self.adata.obs['condition'].isin(test_perts))
                    self.adata.obs.loc[mask, 'mode'] = 'test'
            else:
                self.adata.obs.loc[self.adata.obs['condition'].isin(self.split_results[fold]['test']), 'mode'] = 'test'
            
            # Data-efficiency regime: subsample training perturbations to train_frac (keep all
            # control cells). Fixed seed => baseline & causal see the IDENTICAL reduced train set.
            train_frac = float(kwargs.get('train_frac', 1.0))
            if train_frac < 1.0:
                tr = self.adata.obs['mode'] == 'train'
                nonctrl = [c for c in self.adata.obs.loc[tr & (self.adata.obs['control'] != 1),
                                                         'condition'].unique()]
                rng = np.random.RandomState(int(kwargs.get('subsample_seed', 42)))
                k = max(1, int(round(len(nonctrl) * train_frac)))
                keep = set(rng.choice(np.array(nonctrl), size=k, replace=False).tolist())
                drop = tr & (self.adata.obs['control'] != 1) & (~self.adata.obs['condition'].isin(keep))
                self.adata.obs.loc[drop, 'mode'] = 'excluded'
                print(f"[train_frac={train_frac}] kept {k}/{len(nonctrl)} train perturbations "
                      f"(+all control); dropped {int(drop.sum())} cells")

            self.adata_train = self.adata[self.adata.obs['mode'] == 'train']
            self.adata_test = self.adata[(self.adata.obs['mode'] == 'test') | (self.adata.obs['control'] == 1)]
            
            sc.pp.highly_variable_genes(self.adata_test, inplace=True, n_top_genes=infer_top_gene)
            self.adata_test = self.adata_test[:,self.adata_test.var['highly_variable']]
            
            condition = np.unique(list(self.adata.obs['condition']))
            unique_perturbation = []
            np.array([unique_perturbation.extend(perturbation.split('+')) for perturbation in condition])
            unique_perturbation = np.unique(unique_perturbation)
            unique_perturbation.sort()
            self.unique_perturbation = unique_perturbation
            self.perturbation_dict = {perturbation: i for i, perturbation in enumerate(unique_perturbation)}
            
        else:
            raise ValueError(self.data_name + ' is not a valid data name')
        
        if 'fold' in kwargs.keys():
            fold = kwargs['fold']
        else:
            fold = 0
        if use_negative_edge:
            mask_path = os.path.join(self.data_path, self.data_name,'mask_fold_'+str(fold)+'topk_'+str(k)+split_method+'_negative_edge'+'.pt')
        else:
            mask_path = os.path.join(self.data_path, self.data_name,'mask_fold_'+str(fold)+'topk_'+str(k)+split_method+'.pt')
        if os.path.exists(mask_path):
            self.mask = torch.load(mask_path)
        else:
            # Build co-expression mask from ALL train cells (matches original backbone;
            # norman ~59k dense-es in ~65s). Only subsample for very large datasets
            # (e.g. replogle 250k) to avoid OOM — norman/combosciplex use full cells so
            # the mask is byte-identical to the original codebase (backbone unchanged).
            _Xtr = self.adata_train.X
            if _Xtr.shape[0] > 150000:
                _ridx = np.random.RandomState(0).choice(_Xtr.shape[0], 150000, replace=False)
                _Xtr = _Xtr[_ridx]
            X = _Xtr.toarray() if hasattr(_Xtr, 'toarray') else np.array(_Xtr)
            mask = build_gene_coexpression_graph(X,
                method="pearson",
                wgcna_beta=None,
                sparsify="topk",
                k=k,
                use_negative_edge=use_negative_edge)
            mask = sorted_pad_mask(mask, pad_size=4, gene_names=list(self.adata_train.var_names))
            torch.save(mask, mask_path)
            print('mask saved')
        
        
    def load_flow_data(self, batch_size = 128):
        if self.data_name == 'combosciplex':
            train_sampler = TrainSampler(self.data_name, self.adata_train, ["Drug1", "Drug2"], self.perturbation_dict)
            test_sampler = TestDataset(self.data_name, self.adata_test, ["Drug1", "Drug2"], self.perturbation_dict)
            
            return train_sampler , test_sampler, []
        elif self.data_name in ['norman', 'norman_umi_go_filtered', 'parse_pbmc', 'op3', 'replogle', 'dixit']:
            pair_col = 'donor' if self.data_name == 'parse_pbmc' else None
            train_sampler = TrainSampler(self.data_name, self.adata_train, ["Drug1", "Drug2"],
                                         self.perturbation_dict, pair_col=pair_col)
            test_sampler = TestDataset(self.data_name, self.adata_test, ["Drug1", "Drug2"],
                                       self.perturbation_dict, pair_col=pair_col)
            return train_sampler, test_sampler, []
        else:
            raise ValueError(self.data_name + ' is not a valid data name')
            

    def pretrain_data(self, batch_size = 128):
        if self.data_name == 'combosciplex':
            
            self.pretrain_train_data = PretrainData(self.adata_train, self.perturbation_dict)
            self.pretrain_train_data_loader = DataLoader(self.pretrain_train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
            self.pretrain_test_data = PretrainData(self.adata_test, self.perturbation_dict)
            self.pretrain_test_data_loader = DataLoader(self.pretrain_test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
            return self.pretrain_train_data_loader, self.pretrain_test_data_loader
        else:
            raise ValueError(self.data_name + ' is not a valid data name')
        
class TrainSampler:
    def __init__(self, data_name, adata: sc.AnnData, perturbation_covariates: list[str],
                 perturbation_dict: dict, pair_col: str = None):
        self.data_name = data_name
        self.adata = adata
        self.perturbation_covariates = perturbation_covariates
        self.adata.obs['perturbation_covariates'] = self.adata.obs[perturbation_covariates].apply(lambda x: '+'.join(x), axis=1)
        self._perturbation_covariates = np.asarray(adata.obs['perturbation_covariates'].unique(), dtype=object)

        self._perturbation_covariates = self._perturbation_covariates[self._perturbation_covariates != 'control+control']

        self._perturbation_covariates.sort()
        self.perturbation_covariates_dict = {perturbation: i for i, perturbation in enumerate(self._perturbation_covariates)}

        perturbation_covariates_id = [adata.obs[perturbation_covariates[i]].apply(lambda x: perturbation_dict[x])
                                    for i in range(len(perturbation_covariates))]
        self.perturbation_covariates_id = np.array(perturbation_covariates_id).T

        self.cells_name = self.adata.obs_names

        pc = self.adata.obs['perturbation_covariates'].to_numpy()
        self.control_idx = (pc == 'control+control').nonzero()[0]
        self.idx_by_pert = {p: (pc == p).nonzero()[0] for p in self._perturbation_covariates}

        self.pair_col = pair_col if (pair_col is not None and pair_col in self.adata.obs.columns) else None
        if self.pair_col is not None:
            grp = self.adata.obs[self.pair_col].astype(str).to_numpy()
            self.control_idx_by_group = {}
            for g in np.unique(grp):
                idx = ((pc == 'control+control') & (grp == g)).nonzero()[0]
                if len(idx) > 0:
                    self.control_idx_by_group[g] = idx
            self.idx_by_pert_group = {}
            for p in self._perturbation_covariates:
                pert_mask = pc == p
                per_group = {}
                for g in self.control_idx_by_group:
                    idx = (pert_mask & (grp == g)).nonzero()[0]
                    if len(idx) > 0:
                        per_group[g] = idx
                self.idx_by_pert_group[p] = per_group

    def get_batch(self, batch_size: int, same_perturbation: bool = True):
        if not same_perturbation:
            raise ValueError('same_perturbation must be True')

        perturbation_idx = np.random.choice(len(self._perturbation_covariates), 1)[0]
        perturbation_id = self._perturbation_covariates[perturbation_idx]

        src_idx = self.control_idx
        tgt_idx = self.idx_by_pert[perturbation_id]

        if self.pair_col is not None:
            groups = list(self.idx_by_pert_group.get(perturbation_id, {}).keys())
            if len(groups) > 0:
                sizes = np.array([len(self.idx_by_pert_group[perturbation_id][g]) for g in groups], dtype=float)
                g = groups[np.random.choice(len(groups), p=sizes / sizes.sum())]
                tgt_idx = self.idx_by_pert_group[perturbation_id][g]
                src_idx = self.control_idx_by_group[g]

        tgt_batch_idx = np.random.choice(tgt_idx, batch_size)
        src_batch_idx = np.random.choice(src_idx, batch_size)

        tgt_batch = torch.from_numpy(self.adata.X[tgt_batch_idx].toarray()).float()
        src_batch = torch.from_numpy(self.adata.X[src_batch_idx].toarray()).float()

        return {
            'src_cell_data': src_batch,
            'tgt_cell_data': tgt_batch,
            'src_cell_id': self.cells_name[src_batch_idx],
            'tgt_cell_id': self.cells_name[tgt_batch_idx],
            'condition_id': self.perturbation_covariates_id[tgt_batch_idx],
        }


class TestDataset:
    def __init__(self, data_name, adata: sc.AnnData, perturbation_covariates: list[str],
                 perturbation_dict: dict, pair_col: str = None):
        self.data_name = data_name
        self.adata = adata
        self.perturbation_covariates = perturbation_covariates
        self.adata.obs['perturbation_covariates'] = self.adata.obs[perturbation_covariates].apply(lambda x: '+'.join(x), axis=1)

        self.pair_col = pair_col if (pair_col is not None and pair_col in self.adata.obs.columns) else None
        if self.pair_col is None:
            self.groups = None
            keys = np.asarray(self.adata.obs['perturbation_covariates'].unique(), dtype=object)
            keys = keys[keys != 'control+control']
        else:
            self.groups = self.adata.obs[self.pair_col].astype(str).to_numpy()
            pc = self.adata.obs['perturbation_covariates'].to_numpy()
            non_control = pc != 'control+control'
            keys = np.asarray(np.unique(
                np.array([f'{g}|{p}' for g, p in zip(self.groups[non_control], pc[non_control])], dtype=object)
            ), dtype=object)

        keys.sort()
        self._perturbation_covariates = keys

        n_eval_conditions = int(os.environ.get('SPEAR_N_EVAL_CONDITIONS', '10'))
        if n_eval_conditions is not None and len(self._perturbation_covariates) > n_eval_conditions:
            rng = np.random.default_rng(42)
            self._perturbation_covariates = rng.choice(
                self._perturbation_covariates, size=n_eval_conditions, replace=False
            )
        self.perturbation_covariates_dict = {perturbation: i for i, perturbation in enumerate(self._perturbation_covariates)}

        perturbation_covariates_id = [adata.obs[perturbation_covariates[i]].apply(lambda x: perturbation_dict[x])
                                    for i in range(len(perturbation_covariates))]
        self.perturbation_covariates_id = np.array(perturbation_covariates_id).T

        self.cells_name = self.adata.obs_names

    def _mask(self, key):
        pc = self.adata.obs['perturbation_covariates'].to_numpy()
        if self.pair_col is None:
            return pc == key
        group, perturbation = key.split('|', 1)
        return (pc == perturbation) & (self.groups == group)

    def get_control_data(self, perturbation: str = None):
        mask = self.adata.obs['is_control'].to_numpy()
        if self.pair_col is not None and perturbation is not None:
            group = perturbation.split('|', 1)[0]
            mask = mask & (self.groups == group)
        control_data = self.adata[mask]
        return {
            'src_cell_data': torch.from_numpy(control_data.X.toarray()).float(),
            'src_cell_id': control_data.obs_names,
            'condition_id': torch.tensor(self.perturbation_covariates_id[mask]),
        }

    def get_perturbation_data(self, perturbation: str):
        mask = self._mask(perturbation)
        perturbation_data = self.adata[mask]
        return {
            'tgt_cell_data': torch.from_numpy(perturbation_data.X.toarray()).float(),
            'tgt_cell_id': perturbation_data.obs_names,
            'condition_id': torch.tensor(self.perturbation_covariates_id[mask]),
        }
    

class PerturbationDataset(Dataset):
    def __init__(self, sampler: TrainSampler, batch_size: int, cell_type_col: str = None, 
                 plate_col: str = None, context_to_id: dict = None):
        self.sampler = sampler
        self.batch_size = batch_size
        self.perturbations = sampler._perturbation_covariates
        self.control_idx = (sampler.adata.obs['perturbation_covariates'] == 'control+control').to_numpy().nonzero()[0]

        self.cell_type_col = None
        if cell_type_col is not None and cell_type_col in sampler.adata.obs.columns:
            self.cell_type_col = cell_type_col
            obs = sampler.adata.obs
            control_mask = obs['perturbation_covariates'] == 'control+control'

            self.control_by_celltype = {}
            for ct in obs.loc[control_mask, cell_type_col].unique():
                idx = (control_mask & (obs[cell_type_col] == ct)).to_numpy().nonzero()[0]
                if len(idx) >= 1:
                    self.control_by_celltype[ct] = idx

            self.perturbed_by_pert_celltype = {}
            for pert in self.perturbations:
                pert_mask = obs['perturbation_covariates'] == pert
                self.perturbed_by_pert_celltype[pert] = {}
                for ct in self.control_by_celltype:
                    idx = (pert_mask & (obs[cell_type_col] == ct)).to_numpy().nonzero()[0]
                    if len(idx) >= 1:
                        self.perturbed_by_pert_celltype[pert][ct] = idx

        self.plate_col = None
        if plate_col is not None and plate_col in sampler.adata.obs.columns:
            self.plate_col = plate_col
            obs = sampler.adata.obs
            control_mask = obs['perturbation_covariates'] == 'control+control'

            self.control_by_plate = {}
            for plate in obs.loc[control_mask, plate_col].unique():
                idx = (control_mask & (obs[plate_col] == plate)).to_numpy().nonzero()[0]
                if len(idx) >= 1:
                    self.control_by_plate[plate] = idx

            self.perturbed_by_pert_plate = {}
            for pert in self.perturbations:
                pert_mask = obs['perturbation_covariates'] == pert
                self.perturbed_by_pert_plate[pert] = {}
                for plate in self.control_by_plate:
                    idx = (pert_mask & (obs[plate_col] == plate)).to_numpy().nonzero()[0]
                    if len(idx) >= 1:
                        self.perturbed_by_pert_plate[pert][plate] = idx

        self.context_codes = np.zeros(sampler.adata.n_obs, dtype=np.int64)
        if context_to_id is not None and cell_type_col is not None \
                and cell_type_col in sampler.adata.obs.columns:
            ct_vals = sampler.adata.obs[cell_type_col].astype(str).values
            self.context_codes = np.array(
                [context_to_id.get(v, 0) for v in ct_vals], dtype=np.int64
            )

        _pc = sampler.adata.obs['perturbation_covariates'].to_numpy()
        self.idx_by_pert = {p: (_pc == p).nonzero()[0] for p in self.perturbations}

    def __len__(self):
        return len(self.perturbations) * 1000

    def __getitem__(self, idx):
        perturbation_idx = np.random.choice(len(self.perturbations))
        perturbation_id = self.perturbations[perturbation_idx]

        if self.cell_type_col is not None:
            available_cts = [
                ct for ct in self.perturbed_by_pert_celltype.get(perturbation_id, {})
                if ct in self.control_by_celltype
            ]
            if len(available_cts) > 0:
                ct_sizes = np.array(
                    [len(self.perturbed_by_pert_celltype[perturbation_id][ct]) for ct in available_cts],
                    dtype=float
                )
                ct_probs = ct_sizes / ct_sizes.sum()
                ct_choices = np.random.choice(len(available_cts), self.batch_size, replace=True, p=ct_probs)
                tgt_batch_idx = np.array([
                    np.random.choice(self.perturbed_by_pert_celltype[perturbation_id][available_cts[c]])
                    for c in ct_choices
                ])
                src_batch_idx = np.array([
                    np.random.choice(self.control_by_celltype[available_cts[c]])
                    for c in ct_choices
                ])
            else:
                tgt_idx = self.idx_by_pert[perturbation_id]
                tgt_batch_idx = np.random.choice(tgt_idx, self.batch_size)
                src_batch_idx = np.random.choice(self.control_idx, self.batch_size)
        elif self.plate_col is not None:
            available_plates = [
                p for p in self.perturbed_by_pert_plate.get(perturbation_id, {})
                if p in self.control_by_plate
            ]
            if len(available_plates) > 0:
                plate_sizes = np.array(
                    [len(self.perturbed_by_pert_plate[perturbation_id][p]) for p in available_plates],
                    dtype=float
                )
                plate_probs = plate_sizes / plate_sizes.sum()
                chosen_plate = available_plates[np.random.choice(len(available_plates), p=plate_probs)]
                tgt_batch_idx = np.random.choice(
                    self.perturbed_by_pert_plate[perturbation_id][chosen_plate], self.batch_size
                )
                src_batch_idx = np.random.choice(
                    self.control_by_plate[chosen_plate], self.batch_size
                )
            else:
                # fallback: no plate has both controls and this perturbation
                tgt_idx = self.idx_by_pert[perturbation_id]
                tgt_batch_idx = np.random.choice(tgt_idx, self.batch_size)
                src_batch_idx = np.random.choice(self.control_idx, self.batch_size)
        else:
            tgt_idx = self.idx_by_pert[perturbation_id]
            tgt_batch_idx = np.random.choice(tgt_idx, self.batch_size)
            src_batch_idx = np.random.choice(self.control_idx, self.batch_size)

        X = self.sampler.adata.X
        if hasattr(X[src_batch_idx], "toarray"):
            src_batch = torch.from_numpy(X[src_batch_idx].toarray()).float()
            tgt_batch = torch.from_numpy(X[tgt_batch_idx].toarray()).float()
        else:
            src_batch = torch.from_numpy(np.array(X[src_batch_idx])).float()
            tgt_batch = torch.from_numpy(np.array(X[tgt_batch_idx])).float()

        return {
            'src_cell_data': src_batch,
            'tgt_cell_data': tgt_batch,
            'src_cell_id': list(self.sampler.cells_name[src_batch_idx]),
            'tgt_cell_id': list(self.sampler.cells_name[tgt_batch_idx]),
            'condition_id': torch.tensor(self.sampler.perturbation_covariates_id[tgt_batch_idx]),
            'context_id': torch.from_numpy(self.context_codes[tgt_batch_idx]),
        }


class BinDiscretizer:
    """
    data = np.random.exponential(scale=2.0, size=1000)

    bd = BinDiscretizer(n_bins=200)
    bd.fit(data)

    bd.save_edges('./data/combosciplex/bin_discretizer_edges.pkl')

    new_bd = BinDiscretizer(n_bins=200)
    new_bd.load_edges('./data/combosciplex/bin_discretizer_edges.pkl')

    binned = bd.transform(data)

    recon = bd.inverse_transform(binned, random=False)
    
    Note: 0 is treated as a separate class (class 0), and non-zero values are discretized into classes 1 to n_bins.
    """
    def __init__(self, n_bins: int, strategy: str = "quantile"):
        self.n_bins = n_bins
        self.strategy = strategy
        self.edges = None  # will be (n_bins + 1, ) array

    def fit(self, data: Union[np.ndarray, torch.Tensor]):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        data = data.flatten()
        data = data[data > 0]  # exclude zeros from fitting

        if len(data) == 0:
            raise ValueError("No non-zero entries in data to fit.")

        if self.strategy == "quantile":
            self.edges = np.quantile(data, np.linspace(0, 1, self.n_bins + 1))
        elif self.strategy == "uniform":
            self.edges = np.linspace(data.min(), data.max(), self.n_bins + 1)
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.edges is None:
            raise RuntimeError("Call fit() before transform().")

        is_torch = isinstance(data, torch.Tensor)
        if is_torch:
            orig_dtype = data.dtype
            data = data.detach().cpu().numpy()

        out = np.zeros_like(data, dtype=np.int64)
        mask = data > 0

        # Digitize non-zero entries: 0 remains 0, non-zero values get classes 1 to n_bins
        if np.any(mask):
            # np.digitize returns 0-based indices for the bins
            # We want to map these to 1-based class indices
            digitized = np.digitize(data[mask], self.edges[1:-1])
            # Convert 0-based bin indices to 1-based class indices
            # digitized=0 means it's in the first bin, which should be class 1
            # digitized=1 means it's in the second bin, which should be class 2, etc.
            out[mask] = digitized + 1

        if is_torch:
            return torch.from_numpy(out).to(dtype=torch.int64)
        return out

    def inverse_transform(self, digitized: Union[np.ndarray, torch.Tensor], random: bool = False) -> Union[np.ndarray, torch.Tensor]:
        if self.edges is None:
            raise RuntimeError("Call fit() before inverse_transform().")

        is_torch = isinstance(digitized, torch.Tensor)
        if is_torch:
            orig_dtype = digitized.dtype
            digitized = digitized.detach().cpu().numpy()

        out = np.zeros_like(digitized, dtype=np.float64)
        mask = digitized > 0

        if np.any(mask):
            ids = digitized[mask]
            # Ensure ids are within valid range (1 to n_bins)
            ids = np.clip(ids, 1, self.n_bins)
            # Convert 1-based class indices back to 0-based bin indices
            bin_ids = ids - 1
            lefts = self.edges[bin_ids]
            rights = self.edges[bin_ids + 1]

            if random:
                out[mask] = np.random.uniform(lefts, rights)
            else:
                out[mask] = (lefts + rights) / 2

        if is_torch:
            return torch.from_numpy(out).to(dtype=torch.float64)
        return out
    
    def save_edges(self, filepath: Union[str, Path]):
        """Save edges to a file"""
        if self.edges is None:
            raise RuntimeError("No edges to save. Call fit() first.")
        
        filepath = Path(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump({'edges': self.edges, 'n_bins': self.n_bins}, f)
            
    def load_edges(self, filepath: Union[str, Path]):
        """Load edges from a file"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")
            
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            loaded_n_bins = data['n_bins']
            if loaded_n_bins != self.n_bins:
                raise ValueError(f"Loaded n_bins ({loaded_n_bins}) does not match initialized n_bins ({self.n_bins})")
            self.edges = data['edges']


class PretrainData(Dataset):
    def __init__(self, adata: sc.AnnData, drug_dict: dict):
        self.adata = adata
        self.drug_dict = drug_dict
        self.X = torch.from_numpy(adata.X.toarray()).float()
        self.cell_id = adata.obs_names
        # self.drug1 = torch.tensor(np.array(adata.obs['Drug1'].apply(lambda x: drug_dict[x])))
        # self.drug2 = torch.tensor(np.array(adata.obs['Drug2'].apply(lambda x: drug_dict[x])))
        
    def __len__(self):
        return len(self.adata)
    
    def __getitem__(self, idx):
        return {
            'values' : self.X[idx], 
            'cell_id': self.cell_id[idx],
        }
    
            
class FlowMatchingDataset(Dataset):
    """PyTorch Dataset for flow matching training data"""
    
    def __init__(self, jax_sampler, num_samples=10000, seed=42):
        """
        Args:
            jax_sampler: JAX-based TrainSampler
            num_samples: Number of samples to generate per epoch
            seed: Random seed
        """
        self.jax_sampler = jax_sampler
        self.num_samples = num_samples
        self.rng = jax.random.PRNGKey(seed)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        """Sample a batch from the JAX sampler"""
        # Generate new random key for each sample
        self.rng, sample_key = jax.random.split(self.rng)
        
        # Sample from JAX sampler
        sample = self.jax_sampler.sample(sample_key)
        
        # Convert JAX arrays to PyTorch tensors
        src_cell_data = torch.from_numpy(np.array(sample['src_cell_data'])).float()
        tgt_cell_data = torch.from_numpy(np.array(sample['tgt_cell_data'])).float()
        
        sample['src_cell_id']
        sample['tgt_cell_id']
        # Convert condition embedding if available
        condition_data = None
        if 'condition' in sample:
            condition_data = {
                key: torch.from_numpy(np.array(val)).float()
                for key, val in sample['condition'].items()
            }
        
        # Convert condition_id embedding if available
        condition_id = None
        if 'condition_id' in sample:
            condition_id = torch.from_numpy(np.array(sample['condition_id'])).long()
        
        return {
            'src_cell_data': src_cell_data,
            'tgt_cell_data': tgt_cell_data,
            'condition': condition_data,
            'condition_id': condition_id,
        }
        
if __name__ == "__main__":
    data = Data(data_path='./data')
    data.load_data(
        # data_name='combosciplex'
        # data_name='parse_pbmc'
        data_name='op3'
        )
    data.process_data()