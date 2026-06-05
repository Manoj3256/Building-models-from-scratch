import numpy as np

class Linear_Regression:
    def __init__(self,learning_rate=0.01,iteration=1000):
        self.learning_rate=learning_rate
        self.iteration=iteration
        self.w=0
        self.b=0
        self.loss=[]

    def fit(self,X,y):
        length=len(X)
        for i in range(self.iteration):
            #prediction
            y_pred=self.predict(X) 

            #Gradient Descent
            dw=(-2/length)*np.sum(X*(y-y_pred))
            db=(-2/length)*np.sum(y-y_pred)

            #Updating parameters
            self.w-=self.learning_rate*dw
            self.b-=self.learning_rate*db

            #Loss
            loss=np.mean((y-y_pred)**2)
            self.loss.append(loss)

    def predict(self,X):
        return self.w*X+self.b