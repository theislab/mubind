import matplotlib.pyplot as plt
import seaborn as sns

def filter_contrib_heatmap(filter_contrib_normed):
    contrib_arr = filter_contrib_normed.unsqueeze(dim=0).detach().numpy()

    plt.figure(figsize=(15, .5))
    # only visualize filters with contribution scores above 0.7
    ax = sns.heatmap(contrib_arr,
                     cmap='RdBu_r',
                     vmin=0.66,
                     center=0.66,
                     cbar_kws={"label": "contribution\nscore"})
    ax.set_xlabel('Filter Number',
                  fontsize=10)
    ax.set_title('Contributions by Filters',
                 fontsize=14,
                 y=1.05)
    
def filter_contrib_simple(filter_contrib_normed, A):
    contrib_arr = filter_contrib_normed.unsqueeze(dim=0).detach().numpy()
    sum_A = A.abs().sum(axis=1).detach().numpy()
    contrib = contrib_arr[0]

    plt.subplot(1, 2, 1)
    print(sum_A.shape, contrib.shape)
    plt.scatter(sum_A,  contrib)
    plt.xlabel('sum activities')
    plt.ylabel('contrib scores')
    
    plt.subplot(1, 2, 2)
    # find out how much each filter contributes to the overall result
    plt.plot(contrib * sum_A)
    plt.xlabel('Filter Number')
    plt.ylabel('Contribution Score * Sum of Activities')
    plt.title('Contribution Score * Sum of Activities by Filters')


# all filters that have scores within 75% of the maximum score are visualized as gray, when using vmin and center
# Plot the first heatmap on the first axis
def contrib_heatmaps(contributions_normalized, sum_A_norm, contrib_times_activities, vmins=None, centers=None,
                     cmap='RdBu_r'):
    filter_contrib_normed = contributions_normalized
    contrib_arr = filter_contrib_normed.unsqueeze(dim=0).detach().numpy()
    fig, axs = plt.subplots(3, 1, figsize=(15, 4.5))  # Create a figure with 3 subplots vertically
    # Plot the first heatmap on the first axis
    sns.heatmap(contrib_arr,
                cmap=cmap,
                vmin=vmins[0] if vmins else None,
                center=centers[0] if centers else None,
                cbar_kws={"label": "contribution score"},
                ax=axs[0])

    axs[0].set_xlabel('Filter Number', fontsize=10)
    axs[0].set_title('Contributions by Filters', fontsize=14, y=1.05)

    # Plot the second heatmap on the second axis
    sns.heatmap(sum_A_norm,
                cmap=cmap,
                vmin=vmins[1] if vmins else None,
                center=centers[1] if centers else None,
                cbar_kws={"label": "sum of activities"},
                ax=axs[1])

    axs[1].set_xlabel('Filter Number', fontsize=10)
    axs[1].set_title('Sum of the absolute entries by Filter', fontsize=14, y=1.05)

    # Plot the third heatmap on the third axis
    sns.heatmap(contrib_times_activities,
                cmap=cmap,
                vmin=vmins[2] if vmins else None,
                center=centers[2] if centers else None,
                cbar_kws={"label": "contribution score * sum of activities"},
                ax=axs[2])

    axs[2].set_xlabel('Filter Number', fontsize=10)
    axs[2].set_title('Contribution Score * Sum of Activities by Filters', fontsize=14, y=1.05)

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()

