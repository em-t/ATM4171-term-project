# TODO

### Rough plan:
1. Explore data with PCA and other methods
2. Decide which features to include in the training data (or decide on a few options to compare)
3. Compare classification methods (training data + validation split)
4. Select classification method -> train (full training data) -> classify (fit test data) -> evaluate performance
5. Prepare submissions
    - Take notes and create visualizations along the way
    - Always keep a working submission of the predictions CSV available
    - Put together final report on Monday

## A rough plan:
- Preprocessing:
  - **Standardization** (for methods that require it necessary)
  - **Validation split**
- Exploratory analysis:
  - *Check papers if time/interest?*
  - Clustering methods:
    - K-means
  - **PCA**
  - **Dimensionality reduction** as a result
  - *Resampling, data augmentation, etc., (e.g. using standard deviation)*
- Classification methods (Training data + validation split):
  - **Logistic Regression**
    - Multiple or also Multinomial?
  - *Random forest*
- Method comparison and final method selection
  - **Cross validation**
  - **Test error rate vs. Validation error rate** comparisons
- Produce final model
- Assessment:
  - Metrics used to evaluate term project/kaggle project
    - **Kaggle evaluation metrics**:
      - Binar accuracy
      - Perplexity
      - Multi-class accuracy
    - McNemar's test
  - "Best possible" probability to keep in mind for classification: Bayes decision boundary
- Prepare submissions:
  - **Classification results .csv**
  - Visualizations for:
    - Cross-validation
    - Confusion matrix
  - **Report**
  - Prepare for presentation

### Note:
- **bold** – mandatory
- *italic* – nice-to-have
