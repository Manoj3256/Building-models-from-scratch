import numpy as np

class LogisticRegression():
    def __init__(self,iterations=100,learning_rate=0.02,w=0,b=0):
        self.iterations=iterations
        self.learning_rate=learning_rate
        self.w=w 
        self.b=b
        self.losses=[]
    def fit(self,X,y):
        length=len(X)

        for i in range(self.iterations):
            #prediction
            pred_y=predict(X)

            #Gradient Descent
            w=(1/length)*np.sum((pred_y-y)*X)
            b=(1/length)*np.sum((pred_y-y))
            self.w-=w*self.learning_rate
            self.b-=b*self.learning_rate

            #loss
            loss=(-1/length)*np.sum(y*np.log(pred_y)+(1-y)*np.log(1-pred_y))
            self.losses.append(loss)
    def predict(self,X):
        z=self.w*X+self.b
        return 1/(1+np.exp(-z))

model=LogisticRegression(100,0.06)



            