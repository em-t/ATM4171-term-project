import numpy as np
from sklearn.metrics import accuracy_score, log_loss


def simulate_kaggle_scoring(
    X_test, y_test2, y_test4,
    model_class2,
    model_prob,
    model_class4,
    print_partial_scores=False,
):
    """
    Imitates the Kaggle scoring process.
    The function takes in the test dataset and the three models for binary classification,
    probability estimation, and multiclass classification.
    Rerturns the final score.
    """
    # Calculate scores for each log_model
    
    pred_class2 = model_class2.predict(X_test)
    pred_prob = model_prob.predict_proba(X_test)[:, 1]
    pred_class4 = model_class4.predict(X_test)

    binary_accuracy = accuracy_score(y_test2, pred_class2)
    perplexity = np.exp(log_loss(y_test2, pred_prob))
    multiclass_accuracy = accuracy_score(y_test4, pred_class4)
    total_score = (
        1
        / 3
        * (
            binary_accuracy
            + multiclass_accuracy
            + np.max([0.0, np.min([1.0, 2.0 - perplexity])])
        )
    )

    if print_partial_scores:
        print(f"Binary accuracy: {binary_accuracy:.4f}")
        print(f"Multiclass accuracy: {multiclass_accuracy:.4f}")
        print(f"Model perplexity: {perplexity:.4f}")
    return total_score

def generate_kaggle_output(
    final_test_df,
    model_prob,
    model_class4,
    scaler=None,
):
    """
    Takes in the testing dataframe and models
    Generates the output format for the kaggle competition
    """
    final_test_X = final_test_df.drop(columns=["id"])
    if scaler is not None:
        final_test_X = scaler.transform(final_test_X)
    out_df = final_test_df[["id"]].copy()
    out_df["class4"] = model_class4.predict(final_test_X)
    out_df["p"] = model_prob.predict_proba(final_test_X)[:, 0]
    return out_df