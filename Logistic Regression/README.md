# Logistic Regression From Scratch

## Overview

This project implements Logistic Regression from scratch using NumPy — no Scikit-Learn, no shortcuts. The goal was to understand what actually happens under the hood when a model learns: how the sigmoid squashes a linear output into a probability,how gradient descent nudges the weights step by step, and how binary cross entropy loss measures whether the model is on the right track.

The implementation lives in `model.py` and covers everything from the forward pass to weight updates.

---

## What's implemented

- Sigmoid activation function
- Binary cross entropy loss
- Gradient descent optimizer
- Loss tracking in every epochs

---


**Prediction (forward pass)**

```
z=w*X+b
y_hat=1/(1+e^(-z))    
```

**Loss (binary cross entropy)**

```
L =-(1/n)*sum(y*log(y_hat)+(1-y)*log(1-y_hat) )
```

This is derived from maximum likelihood estimation — we're maximizing the probability of the correct labels, which after taking the log and flipping the sign becomes this minimization problem.

**Gradients**

```
dw=(1/n)*sum((y_hat-y)*X)
db=(1/n)*sum(y_hat-y )
```

**Weight update**

```
w=w-learning_rate*dw
b=b-learning_rate*db
```

---