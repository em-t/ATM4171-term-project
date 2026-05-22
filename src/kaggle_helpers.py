import numpy as np
from sklearn.metrics import accuracy_score, log_loss


def simulate_kaggle_scoring(
    test_df,
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

    # Process the test dataset for scoring
    std_cols = [col for col in test_df.columns if col.endswith(".std")]
    test_df = test_df.drop(columns=std_cols)
    test_df["class2"] = np.where(
        test_df["class4"].eq("nonevent"), "nonevent", "event"
    )
    X_test = test_df.drop(columns=["class4", "class2"])
    y_test2 = test_df["class2"]
    y_test4 = test_df["class4"]

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
):
    """
    Takes in the testing dataframe and models
    Generates the output format for the kaggle competition
    """
    std_cols = [col for col in final_test_df.columns if col.endswith('.std')]
    final_test_X = final_test_df.drop(columns=std_cols)
    final_test_X = final_test_X.drop(columns=["id", "partlybad", "date"])
    

    out_df = final_test_df[["id"]].copy()
    out_df["class4"] = model_class4.predict(final_test_X)
    out_df["p"] = model_prob.predict_proba(final_test_X)[:, 0]
    return out_df