import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split


def learn(data, class_att, smoothing=0):
    model = {}

    apriori = data[class_att].value_counts()
    apriori = apriori / apriori.sum()
    model['_apriori'] = apriori

    for attribute in data.drop(class_att, axis=1).columns:
        if pd.api.types.is_numeric_dtype(data[attribute]): 
            mat_cont = data.groupby(class_att)[attribute].agg(['mean', 'std'])
            model[attribute] = mat_cont
        else:
            mat_cont = pd.crosstab(data[attribute],data[class_att])
            mat_cont = mat_cont+smoothing
            mat_cont = mat_cont / mat_cont.sum(axis=0)
            model[attribute] = mat_cont
    return model


def predict(model, new_instance):
    class_probabilities = {}
    for class_value in model['_apriori'].index:
        probability = 0
        
        for attribute in model:
            if attribute == '_apriori':
                probability = probability +np.log(model['_apriori'][class_value])
            elif 'mean' in model[attribute].columns and 'std' in model[attribute].columns:
                probability = probability +np.log(1 / math.sqrt(2 * np.pi * model[attribute].loc[class_value, 'std']**2)) - ((new_instance[attribute] - model[attribute].loc[class_value, 'mean'])**2 / (2 * model[attribute].loc[class_value, 'std']**2))
            else:
                probability = probability +np.log( model[attribute][class_value][new_instance[attribute]])
        class_probabilities[class_value] = probability

    prediction = max(class_probabilities, key=class_probabilities.get)

    return prediction, class_probabilities

#%% KORISCENJE 
data = pd.read_csv('data/data/drug.csv') 
train, test = train_test_split( data,test_size=0.2,random_state=42)
test = test.reset_index(drop=True)
model = learn(train, 'Drug',5) 
for i in range(len(test)): 
    testiram = test.iloc[i].drop('Drug') 
    prediction, confidence = predict(model, testiram) 
    test.loc[i, 'prediction'] = prediction 
    for klasa in confidence: 
        test.loc[i, 'class=' + klasa] = confidence[klasa]  
print(test)