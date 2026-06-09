# This is Decision Tree Classifier From Scratch

## Project Overview
This project(model.py) implements a Decision Tree Classifier from scratch.

## Objectives
- Understand Decision Tree mathematics
- Implement Gini Impurity and Information Gain
- Build recursive tree splitting from scratch
- Evaluate model with Precision-Recall and Accuracy metrics
- Visualize the decision tree structure

## Mathematical Formulas

**Gini Impurity:**

Gini(node) = 1 - sum(p_i^2)

where p_i = proportion of class i in the node

**Information Gain:**

IG =Entropy(parent)-weighted_avg(Entropy(children))

Entropy(node)=-sum(p_i*log2(p_i))

**Best Split Selection:**

Gini Gain = Gini(parent)- [(n_left/n)*Gini(left)+(n_right/n)*Gini(right)]

**Prediction:**

Traverse tree -> compare feature <= threshold at each node -> reach leaf -> return majority class

## Project Workflow

1. Load dataset
2. Implement Node and DecisionTree classes
3. Train using recursive best-split algorithm
4. Evaluate model (Accuracy, Precision, Recall, F1)
5. Visualize Precision-Recall curve
