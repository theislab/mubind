import numpy as np
import pandas as pd
import itertools
import sklearn.metrics
import scipy
import torch
import torch.optim as topti
import torch.utils.data as tdata
import pickle
import os
from pathlib import Path
import mubind as mb
import itertools
import glob

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


# currently assumes we have trained models using a single set of hyperparams
    # and not the grid approach (i.e. single model per setup)
# pickle file name does not matter as long as it is somewhere under output_dir

def _get_models(outdir_glob, extension='.pkl', device=None, stop_at=None):
    """
    output_dir (str): some parent dir of the output directory
    extension (str): of model files (.h5 or .pkl)
    """
    models = []
    # matches all files with `extension` under `output_dir`
    for next_path in glob.glob(outdir_glob):
        print(next_path)
        p = Path(next_path)
        if extension in p.name:
            with open(p, 'rb') as f:
                models.append(pickle.load(f))
                print(f'Loaded model {p.name}')

                if stop_at is not None and len(models) >= stop_at:
                    break

    print('# of models', len(models))
    return models


def combine_models(output_dir, extension='.pkl', device=None):
    models = _get_models(output_dir, extension=extension, device=device)
    combined_model = mb.models.Multibind(n_rounds=1, datatype='selex').to(device)

    del combined_model.binding_modes.conv_mono[0:]
    del combined_model.binding_modes.conv_di[0:] # to remove Nones from module lists
    for i, model in enumerate(models):
        print(i, model)
        combined_model.binding_modes.conv_mono.extend(model.binding_modes.conv_mono[1:])
        combined_model.binding_modes.conv_di.extend(model.binding_modes.conv_di[1:])
        combined_model.activities.log_activities += model.activities.log_activities

        print(combined_model.selex_module._parameters['log_etas'].shape)
        print(model.selex_module._parameters['log_etas'].shape)
        combined_model.selex_module._parameters['log_etas'] = combined_model.selex_module._parameters['log_etas'].add(model.selex_module._parameters['log_etas'])

    return combined_model


def binding_modes(output_dir, extension='.pkl', device=None, **kwargs):
    """
    output_dir (str): some parent dir of the output directory
    extension (str): of model files (.h5 or .pkl)
    """
    models = _get_models(output_dir, extension=extension, device=device, **kwargs)
    combined_model = mb.models.Multibind(n_rounds=1, datatype='selex').to(device)
    del combined_model.binding_modes.conv_mono[0:]
    del combined_model.binding_modes.conv_di[0:] # to remove Nones from module lists
    for i, model in enumerate(models):
        # print(i, model)
        combined_model.binding_modes.conv_mono.extend(model.binding_modes.conv_mono[1:])
        combined_model.binding_modes.conv_di.extend(model.binding_modes.conv_di[1:])

    return combined_model.binding_modes


def submatrix(m, start, length, flip, filter_neg_weights=True):
    sub_m = m[:, start: start + length]
    if flip:
        sub_m = np.flip(sub_m, [1])
        sub_m = np.flip(sub_m, [0])

    if filter_neg_weights:
        sub_m[sub_m < 0] = 0

    return sub_m


def distances_dataframe(a, b, min_w_sum=3):
    d = []
    min_w = min(a.shape[-1], b.shape[-1])
    lowest_d = -1, -1
    for k in range(5, min_w):
        # print(k)
        for i in range(0, a.shape[-1] - k + 1):
            ai = submatrix(a, i, k, 0)
            ai_sum = ai.sum()
            if ai_sum < min_w_sum:
                continue

            if ai_sum == 0:
                continue
            for j in range(0, b.shape[-1] - k + 1):

                bi = submatrix(b, j, k, 0)
                bi_sum = bi.sum()
                if bi_sum < min_w_sum:
                    continue

                # print(type(ai), type(bi), ai.shape, bi.shape)
                d.append([i, j, k, ai.shape[-1], bi.shape[-1],
                          ai.sum(), bi.sum(), 0, ((bi - ai) ** 2).sum() / bi.shape[-1]])
                if lowest_d[-1] == -1 or d[-1] < lowest_d[-1] or d[-2] < lowest_d[-1]:
                    lowest_d = i, 0, d[-1]

                bi_rev = submatrix(b, j, k, 1)
                # flipped version
                d.append([i, j, k, ai.shape[-1], bi.shape[-1],
                          ai.sum(), bi.sum(), 1, ((bi_rev - ai) ** 2).sum() / bi.shape[-1]])
                if lowest_d[-1] == -1 or d[-1] < lowest_d[-1] or d[-2] < lowest_d[-1]:
                    lowest_d = i, 1, d[-1]

    res = pd.DataFrame(d, columns=['a_start', 'b_start', 'k', 'a_shape', 'b_shape',
                                   'a_sum', 'b_sum', 'b_flip', 'distance']).sort_values('distance')
    return res

def calculate_distances(mono_list):
    res = []
    for a, b in itertools.product(enumerate(mono_list), repeat=2):
        if a[0] > b[0]:
            continue
        df2 = mb.tl.distances_dataframe(a[1], b[1])
        df2['a'] = a[0]
        df2['b'] = b[0]
        res.append(df2)
        df3 = mb.tl.distances_dataframe(b[1], a[1])
        df3['a'] = b[0]
        df3['b'] = a[0]
        res.append(df3)
        # print(res[-1])
    res = pd.concat(res)
    res = pd.concat([res[['a', 'b']], res[[c for c in res if not c in ['a', 'b']]]], axis=1)
    res['id'] = res['a'].astype(str) + '_' + res['b'].astype(str)
    return res

def reduce_filters(binding_modes, plot=False, thr_group=0.01):
    best = None
    iteration_i = 0
    monos = [b.weight for b in binding_modes.conv_mono]
    monos = [m.cpu().detach().numpy().squeeze() for m in monos]

    while True:
        iteration_i += 1
        print('iteration', iteration_i)

        res = calculate_distances([m.copy() for m in monos])

        if plot:
            import seaborn as sns;
            df_min = res.groupby('id').min()
            hm = df_min.pivot('a', 'b', 'distance').fillna(1)
            sns.clustermap(hm, cmap='Reds_r', figsize=[5, 5]) #  annot=hm, fmt='.2f')

        best = res[res['a'] > res['b']].sort_values('distance').drop_duplicates('id')
        best = best[(best['distance'] < thr_group)]
        # define a mask to ignore grouping
        ignore = set()
        mask_ignore = []
        for ri, r in best.iterrows():
            mask_ignore.append(r['a']  in ignore or r['b'] in ignore)
            ignore.add(r['a'])
            ignore.add(r['b'])
        best['ignore'] = mask_ignore
        best = best[~best['ignore']]
        print('# grouping', best.shape)

        if best.shape[0] == 0:
            print('done. No more groups to generate')
            break

        for ri, r in best.iterrows():
            print(r.values)
            a_i, b_i = r['a'], r['b']
            start_a, start_b = r['a_start'], r['b_start']
            length_a, length_b = r['k'], r['k']
            flip_b = r['b_flip']

            m_a, m_b = monos[a_i].squeeze(), monos[b_i].squeeze()
            sub_m1 = mb.tl.submatrix(m_a, start_a, length_a, 0, filter_neg_weights=False)
            sub_m2 = mb.tl.submatrix(m_b, start_b, length_b, flip_b, filter_neg_weights=False)
            # mb.pl.conv_mono(weights_list=[m_a, m_b, sub_m1, sub_m2], n_rows=4, n_cols=1, figsize=[9, 4])

            width_diff = m_a.shape[-1] - m_b.shape[-1]
            if width_diff > 0:
                m_b = np.concatenate([np.zeros((4, width_diff)), m_b], axis=1)
            elif width_diff < 0:
                m_a = np.concatenate([np.zeros((4, -width_diff)), m_a], axis=1)
            assert m_a.shape[-1] == m_b.shape[-1]

            shift = start_b - start_a
            if shift >= 0:
                print('option 1')
                merged_a = np.concatenate([np.zeros((4, shift)), m_a], axis=1)
                merged_b = np.concatenate([m_b, np.zeros((4, shift))], axis=1)
            else:
                print('option 2')
                merged_a = np.concatenate([m_a, np.zeros((4, -shift))], axis=1)
                merged_b = np.concatenate([np.zeros((4, -shift)), m_b], axis=1)

            if flip_b: # flip the b matrix before merging
                merged_b = mb.tl.submatrix(merged_b, 0, merged_b.shape[-1], 1, filter_neg_weights=False)

            print(m_a.shape, m_b.shape, merged_a.shape, merged_b.shape, width_diff, shift)
            merged = (merged_a + merged_b) / 2

            # reduction. Replace a and remove b
            monos[a_i] = merged
            monos[b_i] = None

            if plot:
                mb.pl.conv_mono(weights_list=[m_a, m_b, sub_m1, sub_m2,
                                              merged_a, merged_b, merged], n_rows=7, n_cols=1, figsize=[9, 4], show=True)

        monos = [m for m in monos if m is not None]
        print('# of remaining groups', len(monos))
    return monos


def min_distance(w1, w2):
    df = distances_dataframe(w1, w2)
    if df.shape[0] == 0:
        return np.nan
    return min(df['distance'])

def distances(w1, w2):
    return distances_dataframe(w1, w2)

