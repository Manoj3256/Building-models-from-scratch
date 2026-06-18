import numpy as np

class K_Means():
    def __init__(self,n_clusters=2,iterations=100,random_state=47):
        self.n_clusters=n_clusters
        self.cluster_centers_=[]
        self.random_state=random_state
        self.iterations=iterations

    def fit(self,X):
        X=X.to_numpy()
        np.random.seed(self.random_state)
        index=np.random.choice(len(X),size=self.n_clusters,replace=False)
        points=X[index]
        for _ in range(self.iterations):
            clusters=[[] for _ in range(self.n_clusters)]
            for j in range(len(X)):
                distance=np.zeros(self.n_clusters)
                for c in range(self.n_clusters):
                    distance[c]=np.sqrt((X[j,0]-points[c,0])**2+(X[j,1]-points[c,1])**2 )
                clusters[np.argmin(distance)].append(X[j])
            for c in range(self.n_clusters):
                points[c]=np.mean(clusters[c],axis=0)
        self.cluster_centers_=points
        print(self.cluster_centers_)

model=K_Means(3,100,47)
model.fit(data)