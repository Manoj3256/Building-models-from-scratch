import numpy as np
tree_model=__import__("Decision Tree Classifier.model")

class Random_Forest_Classifier():
    def __init__(self,n_trees,max_depth):
        self.n_trees=n_trees
        self.max_depth=max_depth
        self.trees=[]
    def fit(self,X,y):
        for _ in range(self.n_trees):
            ind=np.random.choice(len(X),size=len(X),replace=True)
            X_sample=X[ind]
            y_sample=y[ind]

            m=tree_model.DecisionTreeClassifier
            tree_i=m.fit(X_sample,y_sample)
            self.trees.append(tree_i)
    def predict(self,X):
        predictions=[]
        for i in range(self.n_trees):
            self.predictions.append(self.trees[i].predict(X))
        pre=np.bincount(self.predictions).argmax()
        return pre

model=Random_Forest_Classifier(25,10)
