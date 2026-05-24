import matplotlib.cm as cmx
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import numpy as np


def get_colormap(unique_labels, cmap="Accent"):
    color_map = plt.get_cmap(cmap)
    z = range(1, len(unique_labels))
    c_norm = mplc.Normalize(vmin=0, vmax=len(unique_labels))
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=color_map)
    return scalar_map


def get_importance_order(loading_x, loading_y):
    """
    Sort features by importance to two principal components.

    Args:
        - loading_x: Loading values for first principle component.
        - loading_y: Loading values for second principle component.
    """
    # Basically just distance, but in the context of the loading plots.
    dist_sq = np.square(loading_x) + np.square(loading_y)
    return dist_sq.argsort()[::-1]


def sort_by_importance(loading_x, loading_y):
    """
    Sort features by importance to two principal components.

    Args:
        - loading_x: Loading values for first principle component.
        - loading_y: Loading values for second principle component.
    """
    idxs = get_importance_order(loading_x, loading_y)

    # Return sorted arrays
    return loading_x[idxs], loading_y[idxs]


def get_features_by_importance(
    features,
    pca_result,
    component_1,
    component_2,
    n_features=10
):
    coeff = np.transpose(pca_result.components_)
    importance_order = get_importance_order(coeff[:, component_1], coeff[:, component_2])
    most_important_features = [features[i] for i in importance_order[:n_features]]
    return most_important_features

    
# Adapted from: https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot
# Function for plotting dataset 
def biplot(
    X,
    y,
    ax,
    title,
    pca_components_by_idx=[0, 1],
    pca_result=None,
    variables=None,
    include_arrows=False,
    n_arrow_variables=10,
    cmap="Accent"
):
    """
    Args:
        - X: np.ndarray
            PCA-transformed data (i.e., result of PCA.transform()).
        - y: pd.DataFrame / pd.Series
            Labels.
        - ax:
            Plot axes.
        - title: str
            Plot title.
        - pca_result:
            Result of PCA.fit(). Needed for drawing arrows.
        - include_arrows: bool
            If True, arrows will be drawn.
    """
    # Select the PCA components to plot, scale
    x_idx = pca_components_by_idx[0]
    y_idx = pca_components_by_idx[1]
    xs = X[:, x_idx]
    ys = X[:, y_idx]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    scaled_x = xs * scalex
    scaled_y = ys * scaley

    labels = y.values.reshape(1, -1)[0]
    unique_labels = list(set(labels))

    scalar_map = get_colormap(unique_labels, cmap=cmap)

    # Add label points
    for i in range(len(unique_labels)):
        indx = labels == unique_labels[i]
        ax.scatter(scaled_x[indx], scaled_y[indx], color=scalar_map.to_rgba(i), alpha=0.5, s=30, edgecolor=(0, 0, 0, 0.5), label=unique_labels[i])
    
    most_important_features = None
    if include_arrows and (pca_result is not None) and (variables is not None):
        coeff = np.transpose(pca_result.components_)
        if n_arrow_variables > coeff.shape[0]:
            n_arrow_variables = coeff.shape[0]

        importance_order = get_importance_order(coeff[:, x_idx], coeff[:, y_idx])
        most_important_features = [variables[i] for i in importance_order[:n_arrow_variables]]

        x_arrow, y_arrow = sort_by_importance(coeff[:, x_idx], coeff[:, y_idx])
        for i in range(n_arrow_variables):
            ax.annotate(variables[i], xytext=(x_arrow[i], y_arrow[i]), xy=(0, 0), arrowprops=dict(facecolor="black", arrowstyle="<-"))

    ax.set_xlabel(f"PC {x_idx + 1}")
    ax.set_ylabel(f"PC {y_idx + 1}")
    ax.legend(loc="upper left")
    if title is not None:
        ax.set_title(title)
    
    return most_important_features


def plot_explained_variance(
    pve,
    ax,
    include_cumulative=True
):
    xvals = [i for i in range(len(pve))]
    ax.plot(xvals, pve, "-o", c="r", markersize=2, label="PVE")
    
    if include_cumulative:
        cum_pve = np.cumsum(pve)
        ax.plot(xvals, cum_pve, "-o", c="b", markersize=2, label="Cumulative PVE")

    ax.grid()
    ax.set_xlabel("PC")
    ax.legend()
