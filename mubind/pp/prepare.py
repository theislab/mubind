from anndata import AnnData


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
