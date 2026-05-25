import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import MaxNLocator
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def run_kmeans(
    df,
    variables,
    n_clusters,
    n_init=300,
    init="random",
    random_state=None
):
    return KMeans(n_clusters=n_clusters, n_init=n_init, init=init, random_state=random_state).fit(df[variables])


def run_k_means_for_range(
    df,
    variables,
    n_min=1,
    n_max=20,
    init="random", # options: "k-means++", "random"
    n_init=300,
    random_state=None
):
    clustering_results = []
    n_clusters = [i for i in range(n_min, n_max + 1)]

    for n in n_clusters:
        kmeans = run_kmeans(df, variables, n, n_init=n_init, init=init, random_state=random_state)
        clustering_results.append(kmeans)

    return clustering_results, n_clusters


def _lineplot_axes(ax, x, y, x_label=None, y_label=None, title=None, line_style="o-"):
    ax.plot(x, y, line_style)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ticklabel_format(style = 'plain')
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)


def plot_kmeans_loss(
    ax,
    clustering_results,
    n_clusters,
    normalize_losses=False,
    x_label="k",
    y_label="k-means loss",
    title=None
):
    # Turn interactive off, because otherwise calling this function will display
    # the plot twice in jupyter, probably because of inner function _lineplot_axes()
    plt.ioff()
    
    losses = [res.inertia_ for res in clustering_results]
    if normalize_losses:
        losses = normalize_array(losses)
    
    _lineplot_axes(ax, n_clusters, losses, x_label=x_label, y_label=y_label, title=title)
    
    # Turn interactive back on
    plt.ion()


def normalize_array(arr):
    normalizer = MinMaxScaler()
        
    normalized = np.array(arr).reshape(-1, 1)
    normalized = normalizer.fit_transform(normalized).flatten()

    return normalized


# Adapted from:
# Source - https://stackoverflow.com/a/71458151
# Posted by Lev Pleshkov, modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-21, License - CC BY-SA 4.0
def run_kmeans_track_losses_forward_centroids(
    df,
    variables,
    n_clusters,
    n_init=300,
    init="random",
    random_state=None
):
    best_loss = None
    iteration_with_best_loss = 0
    centroids_for_best_loss = None
    
    iterations = n_init
    centroids = None

    for i in range(iterations):
        kmeans = KMeans(n_clusters=n_clusters, max_iter=1, n_init=1, init=(centroids if centroids is not None else init), random_state=random_state)
        kmeans.fit(df[variables])
        prev_centroids = centroids
        centroids = kmeans.cluster_centers_

        if (best_loss is None) or (kmeans.inertia_ < best_loss):
            best_loss = kmeans.inertia_
            iteration_with_best_loss = i
            centroids_for_best_loss = prev_centroids

        if ((i + 1) % 20 == 0) and (best_loss is not None):
            print(f"Initialization: {i + 1}. Best loss so far: {best_loss}")

    # Return best loss
    best_kmeans = KMeans(n_clusters=n_clusters, max_iter=1, n_init=1, init=centroids_for_best_loss, random_state=random_state)

    return best_kmeans, best_loss, iteration_with_best_loss + 1


def get_losses_for_n_kmeans(
    df,
    variables,
    n_clusters,
    n_init=300,
    init="random",
    random_state=None
):
    iterations = n_init

    losses = np.empty(n_init, dtype=int)

    for i in range(iterations):
        kmeans = KMeans(n_clusters=n_clusters, max_iter=1, n_init=1, init=init, random_state=random_state).fit(df[variables])
        losses[i] = kmeans.inertia_
    
    return losses


def get_contingency_table_for_clustering_results(
    classes,
    clusters,
    arrange_by_max_diagonal=True
):
    contingency_table = pd.crosstab(classes, clusters)
    row_ind, col_ind = linear_sum_assignment(contingency_table, maximize=True)
    
    if arrange_by_max_diagonal:
        contingency_table = contingency_table[col_ind]

    return contingency_table


def inverse_probability_weighting(var):
    _, counts, inverse = np.unique(var, return_counts=True, return_inverse=True)

    l = len(var)
    
    weights = np.empty(l, dtype=np.float64)
    for i in range(len(inverse)):
        p = inverse[i] / l
        weights[(counts == i)] = 1 / p

    return weights


def get_loss_details_for_n_kmeans(
    df,
    variables,
    n_clusters,
    n_init=300,
    init="random",
    random_state=None
):
    losses = get_losses_for_n_kmeans(
        df=df,
        variables=variables,
        n_clusters=n_clusters,
        n_init=n_init,
        init=init,
        random_state=random_state
    )

    min_loss = losses.min()
    max_loss = losses.max()

    one_percent = int(n_init / 100)
    lowest_loss_idxs = losses.argsort()[:one_percent]

    first_loss_idx = lowest_loss_idxs.min()
    avg_min_loss_idx = np.average(lowest_loss_idxs)

    msg = (f"Min loss: {min_loss}\n"
          f"Max loss: {max_loss}\n"
          f"First iteration that encountered loss belonging in lowest 1st percentile: {first_loss_idx}\n"
          f"Average number of iterations to encounter loss belonging in lowest 1st percentile: {avg_min_loss_idx}\n")

    return losses, msg
