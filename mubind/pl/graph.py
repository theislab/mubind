import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import mubind as mb

def filter_contrib_heatmap(filter_contrib_normed, title, score):
    # Ensure filter_contrib_normed is a PyTorch tensor
    if not isinstance(filter_contrib_normed, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor")

    # Convert the tensor to a numpy array
    contrib_arr = filter_contrib_normed.unsqueeze(dim=0).detach().numpy()

    plt.figure(figsize=(15, .5))
    
    # Flatten the array to compute the 75th percentile value
    sorted_contrib_arr = np.sort(contrib_arr.flatten())
    vmin = sorted_contrib_arr[3 * len(sorted_contrib_arr) // 4]

    ax = sns.heatmap(contrib_arr,
                     cmap="RdBu_r",
                     vmin=vmin,
                     cbar_kws={"label": score})
    ax.set_xlabel('Filter Number', fontsize=10)
    ax.set_title(title, fontsize=14, y=1.05)
    plt.show()
    
def filter_contrib_simple(filter_contrib_normed, A, output_path=None):
    contrib_arr = filter_contrib_normed.unsqueeze(dim=0).detach().numpy()
    sum_A = A.abs().sum(axis=1).detach().numpy()
    sum_A = sum_A / sum_A.max()
    contrib = contrib_arr[0]
    product = sum_A * contrib

    plt.figure(figsize=(5, 2))

    plt.subplot(1, 2, 1)
    sc = plt.scatter(sum_A, contrib, c=product, cmap='viridis')
    plt.xlabel('Sum Activities', fontsize=8)
    plt.ylabel('Contrib Scores', fontsize=8)
    plt.title('Sum Activities \nvs Contrib Scores', fontsize=10)
    cbar = plt.colorbar(sc)
    cbar.set_label('Product of Scores', fontsize=8)
    
    
    plt.subplot(1, 2, 2)
    plt.plot(contrib * sum_A)
    plt.xlabel('Filter Number', fontsize=8)
    plt.ylabel('Contribution Score \n* Sum of Activities', fontsize=8)
    plt.title('Product of Scores\nby Filters', fontsize=10)
    
    plt.subplots_adjust(wspace=0.6)  # Increase the horizontal space between subplots
    plt.tight_layout(pad=1.0)  # Adjust layout to prevent overlap

    if output_path:
        plt.savefig(output_path, format='pdf')
    else:
        plt.show()
    


def contrib_heatmaps(A, C, D, use_hadamard=True,
                     cmap='RdBu_r', save_pdf=False, pdf_path='contrib_heatmaps.pdf'):
    
    # Compute contributions normalized
    _, contributions, max_singular_value = mb.tl.compute_contributions(A, C, D, use_hadamard=use_hadamard)

    # Normalize the data -> not relevant for the plot though apart from scaling
    contributions_normalized = contributions / max_singular_value

    # Compute the sum of the absolute entries by filter
    sum_A = A.abs().sum(axis=1).detach().numpy()
    sum_A_norm = sum_A / sum_A.max()
    
    # Compute the product of contribs and sum_A
    contrib_times_activities = contributions_normalized * sum_A_norm
    
    contrib_arr = contributions_normalized.unsqueeze(dim=0).detach().numpy()
    sum_A_arr = sum_A_norm.reshape(1, -1)  
    contrib_times_act_arr = contrib_times_activities.unsqueeze(dim=0).detach().numpy()

    fig, axs = plt.subplots(3, 1, figsize=(7, 4))  # Adjusted size to be taller

    sns.heatmap(contrib_arr,
                vmin=0.3,
                cmap=cmap,
                ax=axs[0])

    axs[0].set_xlabel('Filter Number', fontsize=8)
    axs[0].set_title('Contributions by Filters', fontsize=10, y=1.05)
    axs[0].tick_params(axis='x', labelsize=8)
    axs[0].tick_params(axis='y', labelsize=12)

    sns.heatmap(sum_A_arr,
                vmax=0.7,
                cmap=cmap,
                ax=axs[1])

    axs[1].set_xlabel('Filter Number', fontsize=8)
    axs[1].set_title('Sum of the absolute entries by Filter', fontsize=10, y=1.05)
    axs[1].tick_params(axis='x', labelsize=8)
    axs[1].tick_params(axis='y', labelsize=8)

    sns.heatmap(contrib_times_act_arr,
                cmap=cmap,
                ax=axs[2])

    axs[2].set_xlabel('Filter Number', fontsize=8)
    axs[2].set_title('Contribution Score * Sum of Activities by Filters', fontsize=10, y=1.05)
    axs[2].tick_params(axis='x', labelsize=8)
    axs[2].tick_params(axis='y', labelsize=8)

    # Attempt to apply tight layout
    try:
        plt.tight_layout(pad=2.0)  # Increase padding to prevent overlapping
    except Exception as e:
        print(f"tight_layout could not be applied: {e}")
    
    if save_pdf:
        fig.savefig(pdf_path, format='pdf')

    plt.show()

    plt.show()