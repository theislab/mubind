import copy
import time
import datetime

import numpy as np
import pandas as pd
import sklearn.metrics
import scipy
import torch
import torch.optim as topti
import torch.utils.data as tdata

import mubind as mb

from torch.profiler import profile, ProfilerActivity

def calculate_enrichment(data, approx=True, cols=[0, 1]):
    data["p0"] = data[cols[0]] / np.sum(data[cols[0]])
    data["p1"] = data[cols[1]] / np.sum(data[cols[1]])
    data["enr"] = data["p1"] / data["p0"]
    if approx:
        data["enr_approx"] = np.where(data["p0"] == 0, data["p1"] / (data["p0"] + 1e-06), data["enr"])
    return data


def create_datasets(data_file):
    # read data and calculate additional columns
    data = pd.read_csv(data_file, sep="\t", header=None)
    data.columns = ["seq", 0, 1]
    data = calculate_enrichment(data)
    # divide in train and test data
    test_dataframe = data.sample(frac=0.001)
    train_dataframe = data.drop(test_dataframe.index)
    test_dataframe.index = range(len(test_dataframe))
    train_dataframe.index = range(len(train_dataframe))
    # create datasets and dataloaders
    train_data = mb.datasets.SelexDataset(data_frame=train_dataframe)
    train_loader = tdata.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    test_data = mb.datasets.SelexDataset(data_frame=test_dataframe)
    test_loader = tdata.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return train_loader, test_loader


def test_network(model, dataloader, device):
    all_seqs = []
    all_targets = np.zeros((len(dataloader.dataset), max(dataloader.dataset.n_rounds)), dtype=np.float32)
    all_preds = np.zeros((len(dataloader.dataset), max(dataloader.dataset.n_rounds)), dtype=np.float32)
    position = 0
    store_rev = dataloader.dataset.store_rev
    with torch.no_grad():  # we don't need gradients in the testing phase
        for i, batch in enumerate(dataloader):
            # Get a batch and potentially send it to GPU memory.
            mononuc = batch["mononuc"].to(device)
            b = batch["batch"].to(device) if "batch" in batch else None
            rounds = batch["rounds"].to(device) if "rounds" in batch else None
            countsum = batch["countsum"].to(device) if "countsum" in batch else None
            seq = batch["seq"] if "seq" in batch else None
            residues = batch["residues"].to(device) if "residues" in batch else None
            if residues is not None and store_rev:
                mononuc_rev = batch["mononuc_rev"].to(device)
                inputs = {"mono": mononuc, "mono_rev": mononuc_rev, "batch": b, "countsum": countsum,
                          "residues": residues}
            elif residues is not None:
                inputs = {"mono": mononuc, "batch": b, "countsum": countsum, "residues": residues}
            elif store_rev:
                mononuc_rev = batch["mononuc_rev"].to(device)
                inputs = {"mono": mononuc, "mono_rev": mononuc_rev, "batch": b, "countsum": countsum}
            else:
                inputs = {"mono": mononuc, "batch": b, "countsum": countsum}

            inputs['scale_countsum'] = model.datatype == 'selex'
            output = model(**inputs)

            output = output.cpu().detach().numpy()
            if len(output.shape) == 1:
                output = output.reshape(output.shape[0], 1)
            target = rounds.cpu().detach().numpy()
            if len(target.shape) == 1:
                target = target.reshape(output.shape[0], 1)

            all_preds[position:(position + len(seq)), :] = output
            all_targets[position:(position + len(seq)), :] = target
            all_seqs.extend(seq)
            position += len(seq)

    return all_seqs, all_targets, all_preds


# if early_stopping is positive, training is stopped if over the length of early_stopping no improvement happened or
# num_epochs is reached.
def optimize_simple(
    model,
    dataloader,
    device,
    optimiser,
    criterion,
    # reconstruction_crit,
    num_epochs=15,
    early_stopping=-1,
    dirichlet_regularization=0,
    exp_max=40,  # if this value is negative, the exponential barrier will not be used.
    log_each=-1,
    verbose=0,
    r2_per_epoch=False,
):
    # global loss_history
    r2_history = []
    loss_history = []
    best_loss = None
    best_epoch = -1
    if verbose != 0:
        print(
            "optimizer: ",
            str(type(optimiser)).split('.')[-1].split('\'>')[0],
            "\ncriterion:",
            str(type(criterion)).split('.')[-1].split('\'>')[0],
            "\n# epochs:",
            num_epochs,
            "\nearly_stopping:",
            early_stopping,
        )

    for f in ["lr", "weight_decay"]:
        if f in optimiser.param_groups[0]:
            if verbose != 0:
                print("%s=" % f, optimiser.param_groups[0][f], end=", ")

    if verbose != 0:
        print("dir weight=", dirichlet_regularization)

    is_LBFGS = "LBFGS" in str(optimiser)

    store_rev = dataloader.dataset.store_rev if not isinstance(dataloader, list) else dataloader[0].dataset.store_rev

    t0 = time.time()
    n_batches = len(list(enumerate(dataloader)))

    # the total number of trials
    n_trials = sum([d.dataset.rounds.shape[0] for d in (dataloader if isinstance(dataloader, list) else [dataloader])])

    for epoch in range(num_epochs):
        running_loss = 0
        running_crit = 0
        running_rec = 0

        # if dataloader is a list of dataloaders, we have to iterate through those
        dataloader_queries = dataloader if isinstance(dataloader, list) else [dataloader]

        # print(len(dataloader_queries))
        for data_i, next_dataloader in enumerate(dataloader_queries):
            # print(data_i, next_dataloader)
            for i, batch in enumerate(next_dataloader):
                # print(i, 'batches out of', n_batches)
                # Get a batch and potentially send it to GPU memory.
                mononuc = batch["mononuc"].to(device)
                b = batch["batch"].to(device) if "batch" in batch else None

                rounds = batch["rounds"].to(device) if "rounds" in batch else None
                if next_dataloader.dataset.use_sparse:
                    rounds = rounds.squeeze(1)

                n_rounds = batch["n_rounds"].to(device) if "n_rounds" in batch else None
                countsum = batch["countsum"].to(device) if "countsum" in batch else None
                residues = batch["residues"].to(device) if "residues" in batch else None
                protein_id = batch["protein_id"].to(device) if "protein_id" in batch else None
                inputs = {"mono": mononuc, "batch": b, "countsum": countsum}
                if store_rev:
                    mononuc_rev = batch["mononuc_rev"].to(device)
                    inputs["mono_rev"] = mononuc_rev
                if residues is not None:
                    inputs["residues"] = residues
                if protein_id is not None:
                    inputs["protein_id"] = protein_id

                loss = None
                if not is_LBFGS:
                    # PyTorch calculates gradients by accumulating contributions to them (useful for
                    # RNNs).  Hence we must manully set them to zero before calculating them.
                    optimiser.zero_grad(set_to_none=None)

                    # outputs, reconstruction = model(inputs)  # Forward pass through the network.
                    outputs = model(**inputs)  # Forward pass through the network.

                    # weight_dist = model.weight_distances_min_k()
                    if dirichlet_regularization == 0:
                        dir_weight = 0
                    else:
                        dir_weight = dirichlet_regularization * model.dirichlet_regularization()
                    # if the dataloader is a list, then we know the output shape directly by rounds
                    if isinstance(dataloader, list):
                        loss = criterion(outputs[:, :rounds.shape[1]], rounds)
                    else:
                        # define a mask to remove items on a rounds specific manner
                        mask = torch.zeros((n_rounds.shape[0], outputs.shape[1]), dtype=torch.bool, device=device)
                        for i in range(mask.shape[1]):
                            mask[:, i] = ~(n_rounds - 1 < i)
                        loss = criterion(outputs[mask], rounds[mask])
                    loss += dir_weight
                    # loss = criterion(outputs, rounds) + 0.01*reconstruct_crit(reconstruction, residues) + dir_weight
                    if exp_max >= 0:
                        loss += model.exp_barrier(exp_max)
                    loss.backward()  # Calculate gradients.
                    optimiser.step()
                    outputs = model(**inputs)  # Forward pass through the network.
                else:
                    def closure():
                        optimiser.zero_grad()
                        # this statement here is mandatory to
                        outputs = model(**inputs)

                        # weight_dist = model.weight_distances_min_k()
                        if dirichlet_regularization == 0:
                            dir_weight = 0
                        else:
                            dir_weight = dirichlet_regularization * model.dirichlet_regularization()

                        # loss = criterion(outputs, rounds) + weight_dist + dir_weight
                        loss = criterion(outputs, rounds) + dir_weight

                        if exp_max >= 0:
                            loss += model.exp_barrier(exp_max)
                        loss.backward()  # retain_graph=True)
                        return loss

                    loss = optimiser.step(closure)  # Step to minimise the loss according to the gradient.

                running_loss += loss.item()
                # running_crit += criterion(outputs, rounds).item()
                # running_rec += reconstruction_crit(reconstruction, residues).item()

        loss_final = running_loss / len(dataloader)
        rec_final = running_rec / len(dataloader)

        if log_each != -1 and epoch > 0 and (epoch % log_each == 0):
            if verbose != 0:
                total_time = time.time() - t0
                time_epoch_1k = (total_time / max(epoch, 1) / n_trials * 1e3)
                print(
                    "Epoch: %2d, Loss: %.6f, " % (epoch + 1, loss_final),
                    "best epoch: %i, " % best_epoch,
                    "secs per epoch: %.3f s, " % ((time.time() - t0) / max(epoch, 1)),
                    "secs epoch*1k trials: %.3fs" % time_epoch_1k
                )

        if best_loss is None or loss_final < best_loss:
            best_loss = loss_final
            best_epoch = epoch
            model.best_model_state = copy.deepcopy(model.state_dict())
            model.best_loss = best_loss

        # print("Epoch: %2d, Loss: %.3f" % (epoch + 1, running_loss / len(train_dataloader)))
        loss_history.append(loss_final)

        if r2_per_epoch:
            r2_history.append(mb.pl.kmer_enrichment(model, dataloader, k=8, show=False))
        # model.crit_history.append(crit_final)
        # model.rec_history.append(rec_final)

        if early_stopping > 0 and epoch >= best_epoch + early_stopping:
            if verbose != 0:
                total_time = time.time() - t0
                time_epoch_1k = (total_time / max(epoch, 1) / n_trials * 1e3)
                print(
                    "Epoch: %2d, Loss: %.4f, " % (epoch + 1, loss_final),
                    "best epoch: %i, " % best_epoch,
                    "secs per epoch: %.3fs, " % ((time.time() - t0) / max(epoch, 1)),
                    "secs epoch*1k trials: %.3fs" % time_epoch_1k
                )
            if verbose != 0:
                print("early stop!")
            break

    # Print if profiling included. Temporarily removed profiling to save memory.
    # print('Profiling epoch:')
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))
    # prof.export_chrome_trace(f'profile_{epoch}.json')
    total_time = time.time() - t0
    model.total_time += total_time
    if verbose:
        print('Final loss: %.10f' % loss_final)
        print(f'Total time (model/function): (%.3fs / %.3fs)' % (model.total_time, total_time))
        print("Time per epoch (model/function): (%.3fs/ %.3fs)" %
              ((model.total_time / max(epoch, 1)), (total_time / max(epoch, 1))))
        print('Time per epoch per 1k trials: %.3fs' % (total_time / max(epoch, 1) / n_trials * 1e3))
        print('Current time:', datetime.datetime.now())

    model.loss_history += loss_history
    model.r2_history += r2_history

def optimize_iterative(
    train,
    device,
    n_kernels=4,
    w=15,
    # min_w=10,
    max_w=20,
    num_epochs=100,
    early_stopping=15,
    log_each=10,
    opt_kernel_shift=True,
    opt_kernel_length=True,
    expand_length_max=3,
    expand_length_step=1,
    show_logo=False,
    optimiser=None,
    criterion=None,
    seed=None,
    init_random=False,
    joint_learning=False,
    ignore_kernel=False,
    lr=0.01,
    weight_decay=0.001,
    stop_at_kernel=None,
    dirichlet_regularization=0,
    verbose=2,
    exp_max=40,
    shift_max=2,
    shift_step=1,
    r2_per_epoch=False,
    **kwargs,
):
    # color for visualization of history
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628"]

    # verbose print declaration
    if verbose:
        def vprint(*args, **kwargs):
            print(*args, **kwargs)
    else:
        vprint = lambda *a, **k: None  # do-nothing function

    vprint("next w", w, type(w))

    if (isinstance(train, list) and 'SelexDataset' in str(type(train[0].dataset))) or 'SelexDataset' in str(type(train.dataset)):
        if criterion is None:
            criterion = mb.tl.PoissonLoss()

        if not isinstance(train, list):
            n_rounds = train.dataset.n_rounds
            n_batches = train.dataset.n_batches
            enr_series = train.dataset.enr_series
        else:
            n_rounds = max([max(t.dataset.n_rounds) for t in train])
            n_batches = len(train)
            enr_series = True

        use_mono = kwargs.get('use_mono', True)
        use_dinuc = kwargs.get('use_dinuc', True)

        dinuc_mode = kwargs.get('dinuc_mode', 'local')

        vprint("# rounds", set(n_rounds) if not isinstance(n_rounds, int) else n_rounds)
        vprint("# rounds", set(n_rounds) if not isinstance(n_rounds, int) else n_rounds)
        vprint('# use_mono', use_mono)
        vprint('# use_dinuc', use_dinuc)
        vprint('# dinuc_mode', dinuc_mode)
        vprint("# batches", n_batches)
        vprint("# kernels", n_kernels)
        vprint("# initial w", w)
        vprint("# enr_series", enr_series)

        model = mb.models.Mubind(
            datatype="selex",
            kernels=[0] + [w] * (n_kernels - 1),
            n_rounds=n_rounds,
            init_random=init_random,
            n_batches=n_batches,
            enr_series=enr_series,
            **kwargs,
        ).to(device)
    elif isinstance(train.dataset, mb.datasets.PBMDataset) or isinstance(train.dataset, mb.datasets.GenomicsDataset):
        if criterion is None:
            criterion = mb.tl.MSELoss()
        if isinstance(train.dataset, mb.datasets.PBMDataset):
            n_proteins = train.dataset.n_proteins
        else:
            n_proteins = train.dataset.n_cells
        vprint("# proteins", n_proteins)

        if joint_learning or n_proteins == 1:
            model = mb.models.Mubind(
                datatype="pbm",
                kernels=[0] + [w] * (n_kernels - 1),
                init_random=init_random,
                n_batches=n_proteins,
                **kwargs,
            ).to(device)
        else:
            bm_generator = mb.models.BMCollection(n_proteins=n_proteins, n_kernels=n_kernels, init_random=init_random)
            model = mb.models.Mubind(
                datatype="pbm",
                init_random=init_random,
                n_proteins=n_proteins,
                bm_generator=bm_generator,
                n_kernels=n_kernels,
                **kwargs,
            ).to(device)
    elif isinstance(train.dataset, mb.datasets.ResiduePBMDataset):
        model = mb.models.Mubind(
            datatype="pbm",
            init_random=init_random,
            bm_generator=mb.models.BMPrediction(num_classes=1, input_size=21, hidden_size=2, num_layers=1,
                                                seq_length=train.dataset.get_max_residue_length()),
            **kwargs,
        ).to(device)
    else:
        assert False  # not implemented yet


    # this sets up the seed at the first position
    if seed is not None:
        # this sets up the seed at the first position
        for i, s, min_w, max_w in seed:
            if s is not None:
                print(i, s)
                model.set_seed(s, i, min=min_w, max=max_w)
        model = model.to(device)

    # step 1) freeze everything before the current binding mode
    for i in range(0, n_kernels):
        vprint("\nKernel to optimize %i" % i)
        vprint("\nFREEZING KERNELS")

        for feat_i in ['mono', 'dinuc']:
            if i == 0 and feat_i == 'dinuc':
                vprint('optimization of dinuc is not necessary for the intercepts (kernel=0). Skip...')
                continue

            vprint('optimizing feature type', feat_i)
            if i != 0:
                if feat_i == 'dinuc' and not use_dinuc:
                    vprint('the optimization of dinucleotide features is skipped...')
                    continue
                elif feat_i == 'mono' and not use_mono:
                    vprint('the optimization of mononucleotide features is skipped...')
                    continue


            # block kernels that we do not require to optimize
            for ki in range(n_kernels):
                mask_mono = (ki == i) and (feat_i == 'mono')
                mask_dinuc = (ki == i) and (feat_i == 'dinuc')
                if verbose != 0:
                    vprint("setting grad status of kernel (mono, dinuc) at %i to (%i, %i)" % (ki, mask_mono, mask_dinuc))

                model.binding_modes.update_grad_mono(ki, mask_mono)
                model.binding_modes.update_grad_di(ki, mask_dinuc)

            print("\n")

            if show_logo:
                vprint("before kernel optimization.")
                mb.pl.plot_activities(model, train)
                mb.pl.conv_mono(model)
                # mb.pl.conv_mono(model, flip=False, log=False)
                mb.pl.conv_di(model, mode='triangle')

            next_lr = lr if not isinstance(lr, list) else lr[i]
            next_weight_decay = weight_decay if not isinstance(weight_decay, list) else weight_decay[i]
            next_early_stopping = early_stopping if not isinstance(early_stopping, list) else early_stopping[i]

            next_optimiser = (
                topti.Adam(model.parameters(), lr=next_lr, weight_decay=next_weight_decay)
                if optimiser is None
                else optimiser(model.parameters(), lr=next_lr)
            )

            # mask kernels to avoid using weights from further steps into early ones.
            if ignore_kernel:

                model.set_ignore_kernel(np.array([0 for i in range(i + 1)] + [1 for i in range(i + 1, n_kernels)]))

            if verbose != 0:
                print("kernels mask", model.get_ignore_kernel())

            # assert False
            optimize_simple(
                model,
                train,
                device,
                next_optimiser,
                criterion,
                num_epochs=num_epochs,
                early_stopping=next_early_stopping,
                log_each=log_each,
                dirichlet_regularization=dirichlet_regularization,
                exp_max=exp_max,
                verbose=verbose,
            )

            # vprint('grad')
            # vprint(model.binding_modes.conv_mono[1].weight.grad)
            # vprint(model.binding_modes.conv_di[1].weight.grad)
            # vprint('')


            model.loss_color += list(np.repeat(colors[i], len(model.loss_history) - len(model.loss_color)))
            # probably here load the state of the best epoch and save
            model.load_state_dict(model.best_model_state)
            # store model parameters and fit for later visualization
            model = copy.deepcopy(model)
            # optimizer for left / right flanks
            best_loss = model.best_loss

            if show_logo:
                print("\n##After kernel opt / before shift optim.")
                mb.pl.plot_activities(model, train)
                mb.pl.conv_mono(model)
                # mb.pl.conv_mono(model, flip=True, log=False)
                mb.pl.conv_di(model, mode='triangle')
                mb.pl.plot_loss(model)

            # print(model_by_k[k_parms].loss_color)
            #######
            # optimize the flanks through +1/-1 shifts
            #######
            if (opt_kernel_shift or opt_kernel_length) and i != 0:
                model = optimize_width_and_length(train,
                                                  model,
                                                  device,
                                                  expand_length_max,
                                                  expand_length_step,
                                                  shift_max,
                                                  shift_step,
                                                  i,
                                                  feat_i=feat_i,
                                                  colors=colors,
                                                  verbose=verbose,
                                                  lr=next_lr,
                                                  weight_decay=next_weight_decay,
                                                  optimiser=optimiser,
                                                  log_each=log_each,
                                                  exp_max=exp_max,
                                                  dirichlet_regularization=dirichlet_regularization,
                                                  early_stopping=next_early_stopping, criterion=criterion,
                                                  show_logo=show_logo,
                                                  n_kernels=n_kernels,
                                                  w=w,
                                                  max_w=max_w,
                                                  num_epochs=num_epochs,
                                                  **kwargs)

            if show_logo:
                vprint("after shift optimz model")
                mb.pl.plot_activities(model, train)
                mb.pl.conv_mono(model)
                # mb.pl.conv_mono(model, log=False)
                mb.pl.conv_di(model, mode='triangle')
                mb.pl.plot_loss(model)
                print("")

            # the first kernel does not require an additional fit.
            if i == 0:
                continue

            vprint("\n\nfinal refinement step (after shift)...")
            vprint("\nunfreezing all layers for final refinement")

            for ki in range(n_kernels):
                vprint("kernel grad (%i) = %i \n" % (ki, True), sep=", ", end="")
                model.update_grad(ki, ki == i)
            vprint("")

            # define the optimizer for final refinement of the model
            next_optimiser = (
                topti.Adam(model.parameters(), lr=next_lr, weight_decay=next_weight_decay)
                if optimiser is None
                else optimiser(model.parameters(), lr=next_lr)
            )
            # mask kernels to avoid using weights from further steps into early ones.
            if ignore_kernel:
                model.set_ignore_kernel(np.array([0 for i in range(i + 1)] + [1 for i in range(i + 1, n_kernels)]))

            vprint("kernels mask", model.get_ignore_kernel())
            vprint("kernels mask", model.get_ignore_kernel())

            # final refinement of weights
            optimize_simple(
                model,
                train,
                device,
                next_optimiser,
                criterion,
                num_epochs=num_epochs,
                early_stopping=next_early_stopping,
                log_each=log_each,
                dirichlet_regularization=dirichlet_regularization,
                verbose=verbose,
            )

            # load the best model after the final refinement
            model.loss_color += list(np.repeat(colors[i], len(model.loss_history) - len(model.loss_color)))
            model.load_state_dict(model.best_model_state)

            if stop_at_kernel is not None and stop_at_kernel == i:
                break

            if show_logo:
                vprint("\n##final motif signal (after final refinement)")
                mb.pl.plot_activities(model, train)
                mb.pl.conv_mono(model)
                mb.pl.conv_di(model, mode='triangle')
                # mb.pl.conv_mono(model, flip=True, log=False)

            vprint('best loss', model.best_loss)

    vprint('\noptimization finished:')
    vprint(f'total time: {model.total_time}s')
    vprint("Time per epoch (total): %.3f s" % (model.total_time / max(num_epochs, 1)))

    return model, model.best_loss

def optimize_width_and_length(train, model, device, expand_length_max, expand_length_step, shift_max, shift_step, i,
                           colors=None, verbose=False, lr=0.01, weight_decay=0.001, optimiser=None, log_each=10, exp_max=40,
                              num_epochs_shift_factor=3,
                           dirichlet_regularization=0, early_stopping=15, criterion=None, show_logo=False, feat_i=None,
                           n_kernels=4, w=15, max_w=20, num_epochs=100, loss_thr_pct=0.005, **kwargs,):
    """
    A variation of the main optimization routine that attempts expanding the kernel of the model at position i, and refines
    the weights and loss in order to find a better convergence.
    """

    # verbose print declaration
    if verbose:
        def vprint(*args, **kwargs):
            print(*args, **kwargs)
    else:
        vprint = lambda *a, **k: None  # do-nothing function

    n_attempts = 0  # to keep a log of overall attempts
    opt_expand_left = range(0, expand_length_max, expand_length_step)
    opt_expand_right = range(0, expand_length_max, expand_length_step)
    opt_shift = list(range(-shift_max, shift_max + 1, shift_step))
    for opt_option_text, opt_option_next in zip(
        ["WIDTH", "SHIFT"], [[opt_expand_left, opt_expand_right, [0]], [[0], [0], opt_shift]]
    ):
        next_loss = None
        loss_diff_pct = 0
        while next_loss is None or (next_loss < best_loss and loss_diff_pct > loss_thr_pct):
            n_attempts += 1

            vprint("\n%s OPTIMIZATION (%s)..." % (opt_option_text, "first" if next_loss is None else "again"), end="")
            vprint("")
            curr_w = model.get_kernel_width(i)
            if curr_w >= max_w:
                if opt_option_text == 'WIDTH':
                    print("Reached maximum w. Stop...")
                    break

            model = copy.deepcopy(model)
            best_loss = model.best_loss
            next_color = colors[-(1 if n_attempts % 2 == 0 else -2)]

            all_options = []

            options = [
                [expand_left, expand_right, shift]
                for expand_left in opt_option_next[0]
                for expand_right in opt_option_next[1]
                for shift in opt_option_next[2]
            ]

            if opt_option_text == 'SHIFT' and False: # include shifts to center weights
                m = torch.tensor(model.get_kernel_weights(i))
                # print(m)
                m[m < 0] = 0
                m = m.reshape(m.shape[-2:])
                col_pos_means = m.mean(axis=0).cpu()
                w = int(m.shape[-1] / 2)
                # print(w)
                col_means = []
                for j in range(m.shape[-1]):
                    a, b = max(j - w, 0), min(j + w, m.shape[-1])
                    ci = m[:, a:b].mean()
                    col_means.append(j)
                pos_max = torch.argmax(torch.tensor(col_means))

                # mb.pl.conv(model)
                shift_center = (w - pos_max).cpu()
                # print(shift)
                # print('adding option', shift_center)
                options = [[0, 0, -shift_center], [0, 0, shift_center]] + options
                # assert False


            print('options to try', options)

            for expand_left, expand_right, shift in options:

                # if abs(expand_left) + abs(expand_right) + abs(shift) == 0:
                #     continue
                # if abs(shift) > 0:  # skip shift for now.
                #     continue
                if curr_w + expand_left + expand_right > max_w:
                    continue

                # print(expand_left, expand_right, shift)
                # assert False

                vprint("next expand left: %i, next expand right: %i, shift: %i"
                       % (expand_left, expand_right, shift))

                model_shift = copy.deepcopy(model)
                model_shift.loss_history = []
                model_shift.r2_history = []
                model_shift.loss_color = []

                mb.tl.optimize_modified_kernel(
                    model_shift,
                    train,
                    kernel_i=i,
                    shift=shift,
                    expand_left=expand_left,
                    expand_right=expand_right,
                    device=device,
                    num_epochs=num_epochs if opt_option_text == 'WIDTH' else num_epochs * num_epochs_shift_factor,
                    early_stopping=early_stopping,
                    log_each=log_each if opt_option_text == 'WIDTH' else log_each * num_epochs_shift_factor, # log_each,
                    update_grad_i=i,
                    feat_i=feat_i,
                    lr=lr,
                    weight_decay=weight_decay,
                    optimiser=optimiser,
                    criterion=criterion,
                    dirichlet_regularization=dirichlet_regularization,
                    exp_max=exp_max,
                    verbose=verbose,
                    **kwargs,
                )
                vprint('')

                model_shift.loss_color += list(np.repeat(next_color, len(model_shift.loss_history)))
                # print('history left', len(model_left.loss_history))
                weight_mono_i = model_shift.binding_modes.conv_mono[i].weight
                pos_w_sum = float(weight_mono_i[weight_mono_i > 0].sum())

                loss_diff_pct = (best_loss - model_shift.best_loss) / best_loss * 100
                all_options.append([expand_left, expand_right, shift, model_shift,
                                    pos_w_sum, weight_mono_i.shape[-1], loss_diff_pct, model_shift.best_loss])
                # print('\n')
                # vprint("after opt.")

                if show_logo:
                    mb.pl.conv_mono(model_shift)
                    mb.pl.conv_di(model_shift, mode='triangle')

            # for shift, model_shift, loss in all_shifts:
            #     print('shift=%i' % shift, 'loss=%.4f' % loss)
            weight_ref_mono_i = model_shift.binding_modes.conv_mono[i].weight
            pos_w_ref_mono_i_sum = float(weight_ref_mono_i[weight_ref_mono_i > 0].sum())


            best = sorted(
                all_options + [[0, 0, 0, model,
                                pos_w_ref_mono_i_sum, weight_ref_mono_i.shape[-1], 0, model.best_loss]],
                key=lambda x: x[-1],
            )
            if verbose != 0:
                print("sorted")
            best_df = pd.DataFrame(best, columns=["expand.left", "expand.right", "shift", "model",
                                                  'pos_w_sum', 'width', "loss_diff_pct", "loss"],
            )
            best_df['last_loss'] = best_loss
            best_df = best_df.sort_values('loss')

            vprint(best_df[[c for c in best_df if c != 'model']])
            # print('\n history len')
            next_expand_left, next_expand_right, next_position, next_model, next_pos_w, w,\
            loss_diff_pct, next_loss = best_df.values[0][:-1]

            print(next_expand_left, next_expand_right, next_position, next_pos_w, w,
                  loss_diff_pct, next_loss)

            if verbose != 0:
                print("action (expand left, expand right, shift): (%i, %i, %i)\n" %
                      (next_expand_left, next_expand_right, next_position))

            if loss_diff_pct >= loss_thr_pct:
                next_model.loss_history = model.loss_history + next_model.loss_history
                next_model.r2_history = model.r2_history + next_model.r2_history
                next_model.loss_color = model.loss_color + next_model.loss_color

            model = copy.deepcopy(next_model)

            if next_expand_left == 0 and next_expand_right == 0 and next_position == 0 and opt_option_text == 'SHIFT':
                print('This was the last iteration. Done with kernel shift optimization...')
                break

    return model

def optimize_modified_kernel(
    model,
    train,
    shift=0,
    expand_left=0,
    expand_right=0,
    device=None,
    num_epochs=500,
    early_stopping=15,
    log_each=-1,
    feat_i='mono',
    update_grad_i=None,
    use_dinuc=False,
    kernel_i=None,
    lr=0.01,
    weight_decay=0.001,
    optimiser=None,
    criterion=None,
    dirichlet_regularization=0,
    exp_max=40,
    verbose=0,
    **kwargs,
):
    assert expand_left >= 0 and expand_right >= 0
    model.modify_kernel(kernel_i, shift, expand_left, expand_right, device)

    # requires grad update
    n_kernels = len(model.binding_modes)
    for ki in range(n_kernels):
        model.binding_modes.update_grad_mono(ki, (ki == update_grad_i) and (feat_i == 'mono'))
        model.binding_modes.update_grad_di(ki, (ki == update_grad_i) and (feat_i == 'dinuc'))

    # finally the optimiser has to be initialized again.
    optimiser = (
        topti.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if optimiser is None
        else optimiser(model.parameters(), lr=lr)
    )

    if criterion is None:
        criterion = mb.tl.PoissonLoss()

    optimize_simple(
        model,
        train,
        device,
        optimiser,
        criterion,
        num_epochs=num_epochs,
        early_stopping=early_stopping,
        log_each=log_each,
        dirichlet_regularization=dirichlet_regularization,
        exp_max=exp_max,
        verbose=verbose,
    )

    return model


def create_simulated_data(motif="GATA", n_batch=None, n_trials=20000, seqlen=100, batch_sizes=10):
    x2, y2 = mb.datasets.simulate_xy(motif, n_trials=n_trials, seqlen=seqlen, max_mismatches=-1, batch=1)

    # print('skip normalizing...')
    # y2 = ((y2 - y2.min()) / (np.max(y2) - np.min(y2))).astype(np.float32)

    # data = pd.DataFrame({'seq': x1, 'enr_approx': y1})
    data = pd.DataFrame({"seq": x2, "target": y2})
    batch = mb.tl.onehot_covar(np.random.random_integers(0, n_batch - 1, len(y2)))

    # assert n_batch == len(batch_sizes)
    batch_mult = np.array(batch_sizes)[(np.argmax(batch, axis=1))]
    # print(batch_mult)
    # df = pd.DataFrame()
    # df['y.before'] = np.array(data['target'])
    # print(np.argmax(batch, axis=1))
    # df['batch'] = np.argmax(batch, axis=1)
    # df['mult'] = batch_mult
    # df['y.after'] = np.array(data['target']) * np.array(batch_mult)
    # print('head...')
    # print(df.head(25))
    # print('tail')
    # print(df.tail(25))

    data["target"] = np.array(data["target"]) * np.array(batch_mult)
    data["is_count_data"] = np.repeat(1, data.shape[0])
    # assert False

    # divide in train and test data -- copied from above, organize differently!
    test_dataframe = data.sample(frac=0.01)
    train_dataframe = data.drop(test_dataframe.index)
    batch_test = np.argmax(batch[test_dataframe.index], axis=1)
    batch_train = np.argmax(batch[train_dataframe.index], axis=1)

    test_dataframe.index = range(len(test_dataframe))
    train_dataframe.index = range(len(train_dataframe))

    n_train = len(train_dataframe.index)
    n_test = len(test_dataframe.index)
    n_train + n_test

    # create datasets and dataloaders
    # print(train_dataframe)
    # print(batch, n_train)
    # print(batch[:n_train])
    train_data = mb.datasets.ChipSeqDataset(data_frame=train_dataframe)
    train_data.batch = batch_train
    train_data.seq = train_dataframe.seq

    train_loader = tdata.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    test_data = mb.datasets.ChipSeqDataset(data_frame=test_dataframe)
    test_data.batch = batch_test
    test_data.seq = test_dataframe.seq

    test_loader = tdata.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return train_loader, test_loader


def create_multi_data(n_chip=100, n_selex=100, n_batch_selex=3):
    # chip seq
    x_chip, y_chip = mb.datasets.gata_remap(n_sample=n_chip) if n_chip > 0 else [[], []]
    df_chip = pd.DataFrame({"seq": x_chip, "target": y_chip})
    df_chip["is_count_data"] = np.repeat(0, df_chip.shape[0])
    df_chip["batch"] = "0_chip"
    # selex
    # seqlen = 10
    # x_selex, y_selex = mb.datasets.simulate_xy('GATA', n_trials=n_selex, seqlen=seqlen, max_mismatches=-1, batch=50)
    # df_selex = pd.DataFrame({'seq': x_selex, 'target': y_selex})
    # df_selex['is_count_data'] = np.repeat(1, df_selex.shape[0])
    # n_batch = n_batch_selex
    # batch sizes. Dataset 2 has many times more reads than Dataset 1
    batch_sizes = [int(5 * 10**i) for i in range(n_batch_selex)]
    train1, test1 = mb.tl.create_simulated_data(
        motif="GATA",
        n_batch=n_batch_selex,
        n_trials=n_selex,
        seqlen=10,
        batch_sizes=batch_sizes,
    )  # multiplier=100)
    # print(train1.dataset.seq, train1.shape)
    # assert False
    df_selex = pd.DataFrame({"seq": train1.dataset.seq, "target": train1.dataset.target})
    df_selex["batch"] = pd.Series(train1.dataset.batch).astype(str).values + "_selex"
    df_selex["is_count_data"] = np.repeat(1, df_selex.shape[0])
    datasets = [df_chip, df_selex]
    data = pd.concat(datasets)

    # pad nucleotides for leaking regions

    data["seqlen"] = data["seq"].str.len()
    # print(data.head())
    # print(data.tail())
    # assert False

    max_seqlen = data["seq"].str.len().max()
    print("max seqlen", max_seqlen)
    max_seqlen = int(max_seqlen)
    data["k"] = "_"
    padded_seq = data["seq"] + data["k"].str.repeat(max_seqlen - data["seqlen"].astype(int))
    data["seq"] = np.where(data["seqlen"] < max_seqlen, padded_seq, data["seq"])
    data["seqlen"] = data["seq"].str.len()
    assert len(set(data["seqlen"])) == 1

    # batches
    # n_batch = len(datasets)
    # batch = []
    # for i, df in enumerate(datasets):
    #     batch.append(np.repeat(i, df.shape[0]))
    # batch = np.array(np.concatenate(batch))
    # data['batch'] = batch
    # data['batch'] -= np.min(data['batch'])

    # print(data.batch.value_counts())
    # assert False

    #     print(data['batch'].value_counts())
    #     print(data.head())

    # print(data.columns)

    # divide in train and test data -- copied from above, organize differently!
    test_dataframe = data.sample(frac=0.01)
    train_dataframe = data.drop(test_dataframe.index)
    batch_test = test_dataframe["batch"]
    batch_train = train_dataframe["batch"]


    #     print(batch_train)
    #     print(batch_train.value_counts())

    test_dataframe.index = range(len(test_dataframe))
    train_dataframe.index = range(len(train_dataframe))
    n_train = len(train_dataframe.index)
    n_test = len(test_dataframe.index)
    n_train + n_test

    train_data = mb.datasets.MultiDataset(data_frame=train_dataframe)
    train_data.batch = np.array(batch_train)
    train_data.batch_one_hot = mb.tl.onehot_covar(train_data.batch)
    train_loader = tdata.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    # print(np.bincount(batch))

    test_data = mb.datasets.MultiDataset(data_frame=test_dataframe)
    test_data.batch = np.array(batch_test)
    test_data.batch_one_hot = mb.tl.onehot_covar(test_data.batch)
    test_loader = tdata.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    return train_loader, test_loader

def scores(model, train, by=None, **kwargs):
    counts = mb.tl.kmer_enrichment(model, train, **kwargs)

    if by is not None and by == 'batch':
        results_by_batch = {}
        for batch_id, grp in counts.groupby('batch'):
            counts_batch = counts[counts['batch'] == batch_id]
            targets = counts_batch[[c for c in counts_batch if c.startswith('t')]]
            pred = counts_batch[[c for c in counts_batch if c.startswith('p')]]
            mask = np.isnan(targets)

            r2_counts = sklearn.metrics.r2_score(targets.to_numpy()[~mask], pred.to_numpy()[~mask])
            # print(counts)
            r2_enr = sklearn.metrics.r2_score(counts_batch["enr_obs"], counts_batch["enr_pred"])
            try:
                r2_f = sklearn.metrics.r2_score(counts_batch["f_obs"], counts_batch["f_pred"])
            except:
                r2_f = np.nan
            try:
                r_f = scipy.stats.pearsonr(counts_batch["f_obs"], counts_batch["f_pred"])[0]
            except:
                r_f = np.nan
            result = {'r2_counts': r2_counts,
                    'r2_foldchange': r2_f, 'r2_enr': r2_enr,
                    'r2_fc': r_f ** 2,
                    'pearson_foldchange': r_f}
            results_by_batch[batch_id] = result
        return results_by_batch

    else:
        targets = counts[[c for c in counts if c.startswith('t')]]
        pred = counts[[c for c in counts if c.startswith('p')]]
        mask = np.isnan(targets)

        # print(targets[:5])
        # print(pred[:5])
        # assert False
        r2_counts = sklearn.metrics.r2_score(targets.to_numpy()[~mask], pred.to_numpy()[~mask])
        # print(counts)
        r2_enr = sklearn.metrics.r2_score(counts["enr_obs"], counts["enr_pred"])
        try:
            r2_f = sklearn.metrics.r2_score(counts["f_obs"], counts["f_pred"])
        except:
            r2_f = np.nan
        try:
            r_f = scipy.stats.pearsonr(counts["f_obs"], counts["f_pred"])[0]
        except:
            r_f = np.nan
        return {'r2_counts': r2_counts,
                'r2_foldchange': r2_f, 'r2_enr': r2_enr,
                'r2_fc': r_f ** 2,
                'pearson_foldchange': r_f}

def kmer_enrichment(model, train, k=None, base_round=0, enr_round=-1, pseudo_count=1):
    # getting the targets and predictions from the model
    seqs, targets, pred = mb.tl.test_network(model, train, next(model.parameters()).device)
    counts = None
    target_labels = ["t" + str(i) for i in range(max(train.dataset.n_rounds))]
    pred_labels = ["p" + str(i) for i in range(max(train.dataset.n_rounds))]

    if k is not None:
        target_kmers = mb.tl.seqs2kmers(seqs, k=k, counts=targets)
        # print(target_kmers.shape)
        # print(target_kmers.head())
        # print(np.stack(target_kmers["counts"].to_numpy()))

        target_kmers[target_labels] = np.stack(target_kmers["counts"].to_numpy())
        pred_kmers = mb.tl.seqs2kmers(seqs, k=k, counts=pred)
        pred_kmers[pred_labels] = np.stack(pred_kmers["counts"].to_numpy())

        counts = (
            target_kmers[target_labels]
            .merge(pred_kmers[pred_labels], left_index=True, right_index=True, how="outer")
            .fillna(0)
        )
    else:
        t = pd.DataFrame(targets, index = seqs,
                         columns=target_labels)
        p = pd.DataFrame(pred, index = seqs,
                         columns=pred_labels)
        for i in range(max(train.dataset.n_rounds)):
            # print(train.dataset.n_rounds - 1)
            p['p' + str(i)] = np.where(~(train.dataset.n_rounds - 1 < i), p['p' + str(i)], np.nan)
            t['t' + str(i)] = np.where(~(train.dataset.n_rounds - 1 < i), t['t' + str(i)], np.nan)
        counts = pd.concat([t, p], axis=1)
        # print(counts)
        # assert False

        if train.dataset.batch is not None:
            counts['batch'] = train.dataset.batch
        counts['n_rounds'] = train.dataset.n_rounds

    if model.datatype == 'selex':
        if enr_round == -1:
            enr_round = max(train.dataset.n_rounds) - 1

        # print(enr_round)
        # print((pseudo_count + counts[pred_labels[train.dataset.n_rounds]]))
        # assert False
        # print('iterative assignment of scores in last round...')
        # last_target = (~pd.isnull(counts[counts.columns[:9]])).sum(axis=1).values
        target_last_round = np.nan
        pred_last_round = np.nan
        for c in counts.columns:
            if c.startswith('p'):
                pred_last_round = np.where(~pd.isnull(counts[c]), counts[c], pred_last_round)
            if c.startswith('t'):
                target_last_round = np.where(~pd.isnull(counts[c]), counts[c], target_last_round)

        # label_last_round = ('p' + (counts['n_rounds'] - 1).astype(str))
        # pred_last_round = np.array([r[label_last_round[ri]] for ri, r in counts.iterrows()])
        # label_last_round = ('t' + (counts['n_rounds'] - 1).astype(str))
        # target_last_round = np.array([r[label_last_round[ri]] for ri, r in counts.iterrows()])

        counts["enr_pred"] = (pseudo_count + pred_last_round) / (pseudo_count + counts[pred_labels[base_round]])
        # counts["enr_pred"] = (pseudo_count + counts[pred_labels[enr_round]]) / (pseudo_count + counts[pred_labels[base_round]])
        counts["enr_obs"] = (pseudo_count + target_last_round) / (pseudo_count + counts[target_labels[base_round]])
        # counts["enr_obs"] = (pseudo_count + counts[target_labels[enr_round]]) / (pseudo_count + counts[target_labels[base_round]])

        counts["f_pred"] = (1 / (enr_round - base_round)) * np.log10(counts["enr_pred"])
        counts["f_obs"] = (1 / (enr_round - base_round)) * np.log10(counts["enr_obs"])
    elif model.datatype == 'pbm':  # assuming only one column of numbers to be modeled
        counts["enr_pred"] = counts['p0']
        counts["enr_obs"] = counts['t0']
        counts["f_pred"] = counts['p0']
        counts["f_obs"] = counts['t0']
    else:
        assert False

    return counts

def predict(model, train, show=True):
    counts = mb.tl.kmer_enrichment(model, train)
    return counts

def dynamic_score(model):
    tspa = torch.sparse_coo_tensor
    t = torch.transpose
    # connectivities
    C = model.graph_module.conn_sparse
    a_ind = C.indices()
    log_dynamic = model.graph_module.log_dynamic
    D = model.graph_module.log_dynamic
    D_tril = tspa(a_ind, D, C.shape)  # .requires_grad_(True).cuda()
    D_triu = tspa(a_ind, -D, C.shape)  # .requires_grad_(True).cuda()
    D = D_tril + t(D_triu, 0, 1)

    torch.set_printoptions(precision=2)
    dynamic_score = D.to_dense().detach().cpu().sum(axis=0)
    return dynamic_score
