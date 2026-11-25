import pandas as pd
import numpy as np


def learn(data, target, lam=0, stohastic = False):
    Y=data[[target]]
    X=data.drop(target, axis=1)
    x_mean = X.mean()
    x_std = X.std()
    X = (X - x_mean)/x_std
    X["X0"] = 1
    m,n = X.shape
    X = X.to_numpy()
    Y = Y.to_numpy()
    w = np.random.random((1,n))
    alpha= 0.5
    
    for it in range(100000):
        if stohastic:
            i = np.random.randint(0,m)
            prediction = X[i].dot(w.T)
            err = prediction-Y[i]
            regularization = 2 * lam * w
            regularization[0,-1] = 0
            grad = err*X[i] + regularization
        else:
            prediction = X.dot(w.T)
            err = prediction-Y 
            regularization = 2 * lam * w
            regularization[0,-1] = 0
            grad = err.T.dot(X)/m + regularization
            
        w = w - alpha*grad
        MSE = err.T.dot(err) / m 
        
        grad_norm = abs(grad).sum() 
        print(it, grad_norm, MSE, alpha)
        if grad_norm < 0.01: break
    return w, x_mean, x_std;

def predict(model, data, x_mean, x_std):
    data = (data - x_mean)/x_std
    data["X0"]=1
    prediction = data.dot(model.T)
    print(prediction)
    
podaci = pd.read_csv("data/house.csv")
model, x_mean, x_std = learn(podaci, "Price")
data= pd.read_csv("data/house_new.csv")
predict(model, data, x_mean, x_std)
    


