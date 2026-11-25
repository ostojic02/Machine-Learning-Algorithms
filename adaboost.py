import pandas as pd
import numpy as np
import math
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def learn(x,y, ensemble_size, algorithms, learning_rate=1):
    n,m = x.shape
    alfas = pd.Series(np.array([1/n]*n), index=x.index,)
    ensemble = []
    weights = np.zeros(ensemble_size)
    for t in range(ensemble_size):
        alg = np.random.choice(algorithms)
        model = clone(alg).fit(x,y, sample_weight=alfas)
        predictions = model.predict(x)
        error = (predictions!=y).astype(int)
        weighted_error = (error*alfas).sum()
        w = learning_rate*1/2 * math.log((1-weighted_error)/weighted_error)
        ensemble.append(model)
        weights[t] = w
        factor = np.exp(-w*predictions*y)
        alfas = alfas * factor
        z = alfas.sum()
        alfas = alfas/z
        
    return ensemble, weights
        
def predict(x, ensemble, weights):
    predictions = pd.DataFrame([model.predict(x) for model in ensemble]).T
    margin = predictions.dot(weights)
    prediction = np.sign(margin)
    probability_positive = 1 / (1 + np.exp(-2 * margin))
    probability_negative = 1 - probability_positive
    probabilities = pd.DataFrame({'postive':probability_positive, 'negative':probability_negative})
    return prediction, probabilities

data = pd.read_csv('data/data/drugY.csv')
X = data.drop('Drug',axis=1)
X = pd.get_dummies(X)
y = data['Drug']*2-1
algoritmi = [GaussianNB(),DecisionTreeClassifier(max_depth=1)]
ensemble, weights = learn(X, y, 1000, [GaussianNB(), DecisionTreeClassifier(max_depth=1)],
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          
                          )
predictions, verovatnoce = predict(X, ensemble, weights)
accuracy = (predictions == y).mean()
print(accuracy)
