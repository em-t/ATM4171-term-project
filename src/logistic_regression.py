import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Based on course example solutions
def run_lasso_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    # X_test_lasso: pd.DataFrame,
    # y_test: pd.DataFrame
):
    """
    Return fitted lasso logistic regression model
    """
    # X_train_lasso = df_train_standardized[numeric_vars]
    # y_train = df_train_standardized["class2"]
    # X_test_lasso = df_validation_standardized[numeric_vars]
    # y_test = df_validation_standardized["class2"]

    # Lambda conversion
    lambda_glmnet = 0.01
    n = X_train.shape[0]
    C = 1.0 / (lambda_glmnet * n)

    # Lasso logistic regression
    lasso = LogisticRegression(
        l1_ratio=1, # True Lasso regression
        solver="saga", # NOTE: the solver depends on the penalty chosen, see scikit doc.
        C=C, # Inverse of regularizatio strength
        fit_intercept=True,
        max_iter=20000
    )

    lasso.fit(X_train, y_train)

    return lasso


# Based on course example solutions
def model_predict_binary(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    target_label
):
    # Predictions
    p_train_lasso = model.predict_proba(X_train)[:, 1]
    p_test_lasso = model.predict_proba(X_test)[:, 1]

    yhat_train_lasso = (p_train_lasso >= 0.5)
    yhat_test_lasso = (p_test_lasso >= 0.5)

    train_accuracy = np.mean(yhat_train_lasso == (y_train == target_label).values)
    test_accuracy = np.mean(yhat_test_lasso == (y_test == target_label).values)
    # Or, for generic version:
    # pred_test_lasso = model.predict(X_test_lasso)
    # test_accuracy = accuracy_score(pred_class2_lasso, y_test.values)

    return train_accuracy, test_accuracy


# Based on course example solutions
def plot_sigmoid(
    ax,
    model,
    X_train,
    y_train,
    target_label,
    inverse_label,
    title="Training data"
):
    decision_train = model.decision_function(X_train)

    # sigmoid probabilities from decision function
    sigmoid = 1 / (1 + np.exp(-decision_train))

    # smooth logistic curve
    x_curve = np.linspace(
        decision_train.min(),
        decision_train.max(),
        500
    )
    y_curve = 1 / (1 + np.exp(-x_curve))

    legend_label = rf"$P(Y =$ {target_label} | $\beta^Tx)$"
    ax.plot(x_curve, y_curve, color="black", label=legend_label)
    ax.axvline(0, linestyle="dotted")
    ax.axhline(0.5, linestyle="dotted", label="decision boundary")

    ax.scatter(
        decision_train[y_train == target_label],
        sigmoid[y_train == target_label],
        color="red",
        marker="o",
        alpha=0.7,
        label=target_label
    )

    ax.scatter(
        decision_train[y_train != target_label],
        sigmoid[y_train != target_label],
        color="blue",
        marker="x",
        alpha=0.7,
        label=inverse_label
    )

    y_label = rf"$P(Y=$ {target_label}$)$"
    ax.set_xlabel(r"$\beta^T x$")
    ax.set_ylabel(y_label)
    ax.legend()
    if title is not None:
        ax.set_title(title)
