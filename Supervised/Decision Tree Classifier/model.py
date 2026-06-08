class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value

class DecisionTreeClassifier():
    def __init__(self,max_depth=None,random_state=None):
        self.max_depth=max_depth
        self.random_state=random_state
    def finding_best_feature(X,y):
        best_feature=None
        best_threshold=None
        best_gini=float('inf')
        for fe in range(X.shape[1]):
            thres=np.unique(X[:,fe])
            for th in thres:
                left_root=X[:,fe]<=th
                right_root=X[:,fe]>th
                if len(left_root) == 0 or len(right_root) == 0:
                    continue
                n = len(y)
                gain=base_gini-(len(left)/n*self.gini(left)+len(right)/n*self.gini(right))
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh
        return best_feat, best_thresh
 
    def gini(self, y):
        classes,counts =np.unique(y,return_counts=True)
        p=counts/len(y)
        return 1-np.sum(p **2)

    def building(self,X,y,depth):
        if self.depth>self.max_depth:
            return 
        best_feature,best_threshold=self.finding_best_feature(X,y)
        left_root=X[:,best_feature]<=best_threshold
        right_root=X[:,best_feature]>best_threshold
        left_subtree=self.building(X[left_root],y[left_root],depth+1)
        right_subtree=self.building(X[right_root],y[right_root],depth+1)
        return Node(feature=best_feature,threshold=best_threshold,left=left_subtree,right=right_subtree)
    
    def fit(self,X,y):
        return self.building(X,y,depth=0)
    
    def predict_one(self, node, x):
        if node.value is not None:
            return node.value
        if x[node.feature]<=node.threshold:
            return self.predict_one(node.left,x)
        return self.predict_one(node.right,x)

    def predict(self,X):
        return np.array([self.predict_one(self.root,x) for x in X])