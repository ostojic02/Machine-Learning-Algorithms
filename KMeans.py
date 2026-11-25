import pandas as pd
import numpy as np


def initializeCentroids(data, k):
    n, m = data.shape
    centroids = data.sample(1).reset_index(drop=True)

    while len(centroids) < k:
        min_distances = []
        for i in range(n):
            distances = ((centroids - data.iloc[i]) ** 2).sum(axis=1)
            min_distances.append(distances.min())

        min_distances = np.array(min_distances)
        probabilities = min_distances / min_distances.sum()
        index = np.random.choice(n, p=probabilities)

        centroids = pd.concat([centroids, data.iloc[[index]]], ignore_index=True)

    return centroids

def validateClusters(data, assignments, centroids, w, max_distance=4, min_centroid_distance=1):
    k = len(centroids)
    average_distances = np.zeros(k)

    for c in range(k):
        subset = data[assignments == c]
        distances = np.sqrt((((subset - centroids.iloc[c]) ** 2) * w).sum(axis=1))
        average_distances[c] = distances.mean()

        if average_distances[c] > max_distance:
            print(f"Centroid klastera {c} loše reprezentuje klaster")

    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            center_dist = np.sqrt((((centroids.iloc[c1] - centroids.iloc[c2]) ** 2) * w).sum())

            if center_dist < min_centroid_distance:
                print(f"Klasteri {c1} i {c2} su previše slični")

    return average_distances
    

def learn(data, w = None, k=5, p=1, randomlyCentroids=True):
    n, m = data.shape
    #normalizacija
    data_mean = data.mean()
    data_std = data.std()
    data = (data - data_mean)/data_std
    if w is None: w= np.ones(m)
    best_quality = float('inf')
    best_centroids = data.sample(k).reset_index(drop=True)
    for j in range(p):
        
        if randomlyCentroids: centoirds = data.sample(k).reset_index(drop=True)
        else:centoirds = initializeCentroids(data, k)
        assign = np.zeros((n,1))
        old_quality = float('inf')
        for iteration in range(50):
            quality = np.zeros(k)
            for i in range(n):
                slucaj = data.iloc[i]
                dist = (((slucaj - centoirds)**2)*w).sum(axis=1)
                assign[i] = np.argmin(dist)
                
            for c in range(k):
                subset = data[assign==c]
                centoirds.iloc[c] = subset.mean()
                quality[c] = (subset.var()*w).sum()*len(subset)
                
            total_quality = quality.sum()
            if(old_quality==total_quality): break
            old_quality = total_quality
            
        centoirds = centoirds*data_std + data_mean
        if(best_quality>old_quality):
            best_quality = old_quality
            best_centroids = centoirds
        
    print(best_centroids)
    return best_centroids
    
def predict(data,centroids):
    n, m = data.shape
    assign = np.zeros(n, dtype=int)
    for i in range(n):
        instance = data.iloc[i]
        dist = ((instance - centroids) ** 2).sum(axis=1)
        assign[i] = np.argmin(dist)
    return assign
 
       
podaci = pd.read_csv("data/data/boston.csv")    
centroids = learn(podaci, p=3)       
 
