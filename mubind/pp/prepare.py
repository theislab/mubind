from anndata import AnnData
import pandas as pd
import numpy as np

def prepare_selex(reads_zero, reads_tf):
    data_by_k_r0 = {}
    for k_r0 in reads_zero:
        print(k_r0)
        next_data = reads_zero[k_r0].copy()
        new_cols = ['seq', k_r0]
        for k_tf in reads_tf:
            next_data = next_data.merge(reads_tf[k_tf], on='seq', how='outer').fillna(
                0)  # .astype(int)
            new_cols.append(k_tf)
        next_data.columns = new_cols
        for i, k in enumerate(new_cols[1:]):
            next_data[k] = next_data[k].astype(int)
            # next_data[i] = next_data[k].astype(int)
        next_data = next_data.set_index('seq')
        next_data['batch'] = 1
        next_data['is_count_data'] = 1
        data_by_k_r0[k_r0] = next_data

    return data_by_k_r0


def sample_rounds(df, n_rounds, n_sample_col, step=10):
    '''
    Iterate through the main rounds and get up to n_sample_col elements, using the measured enrichments versus r0 to
    select groups
    '''
    idx_final = set()
    for c in df.columns[:n_rounds]:
        print(c)

        fg = df[c]
        bg = df[0]

        mask = (fg == 0) & (bg == 0)
        enr = ((fg + 1) / (bg + 1)).sort_values(ascending=False)

        # ignore zero rows
        print(enr.shape)
        print(mask.shape)
        enr = enr[~mask]
        print(enr.shape)

        print()

        categories = enr.value_counts().index

        # print(enr.value_counts())
        print('# categories', len(categories))
        n_sample_cat = int(n_sample_col / len(categories))
        idx_all = enr.index

        n_total = -1
        idx_round = None
        idx_counts_ascending = enr.value_counts().sort_values(ascending=True).index
        while n_total < n_sample_col:
            idx_round = set()
            n_total = 0
            n_sample_cat += step
            for cat in idx_counts_ascending:
                idx_next = idx_all[enr == cat]
                idx_sample = pd.Index(np.random.choice(idx_next, min(n_sample_cat, idx_next.shape[0])))
                # print(cat, idx_sample.shape[0])
                n_total += idx_sample.shape[0]
                idx_round = idx_round.union(set(idx_sample))
            print('# up to %i by category, # total: %i' % (n_sample_cat, n_total))

        # add the selected entries to the rest of the tree
        print(len(idx_round))
        idx_final = idx_final.union(idx_round)
        print(len(idx_final))

    return df[df.index.isin(idx_final)]

def sample_anndata(adata, n_sample_obs = 750, n_sample_var = 750):
    obs_sample = pd.Series(adata.obs_names).sample(n_sample_obs)
    var_sample = pd.Series(adata.var_names).sample(n_sample_var)

    ad = adata[adata.obs_names.isin(obs_sample), adata.var_names.isin(var_sample)].copy()
    return ad
