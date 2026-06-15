# Random Forest Classifier (From Scratch)

A NumPy-only implementation of a Random Forest Classifier, built as an ensemble of Decision Tree Classifiers using **bagging ** and **majority-vote prediction**.

---

##  Overview

A Random Forest is an ensemble learning method that builds multiple decision trees on bootstrapped samples of the training data and combines their predictions through majority voting (for classification). The randomness introduced through bootstrapping reduces variance and helps prevent the overfitting that a single decision tree is prone to.

This implementation reuses the `DecisionTreeClassifier` from the `Decision Tree Classifier` module as the base estimator for each tree in the forest.

---

##  Intuition

A single decision tree tends to overfit — it memorizes the training data, especially when grown deep. Random Forest addresses this by:

1. **Bagging ** — training each tree on a random sample of the original dataset, so each tree sees a slightly different version of the data.
2. **Aggregation** — combining the predictions of all trees via majority vote, which smooths out the individual errors of any one tree.

The result is a model that is typically more accurate and more stable than any individual tree in the ensemble.

---

##  Usage

```python
from model import Random_Forest_Classifier

# Initialize the model
model = Random_Forest_Classifier(n_trees=25, max_depth=10)

model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

##  Files Structure

```
Random Forest Classifier/
├── model.py        
├── notebook.ipynb  
└── README.md       
```

