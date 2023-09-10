import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:

    def __init__(self, k=2, n=2):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.k = k
        self.n = n
        self.centroids = np.zeros(shape=[k, n])

    def fit(self, X, max_it=100, threshold=.01, satisfaction=.2):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            max_it (int): maximum iterations for fitting centroids
            threshold (float): convergence satisfaction in percent
            satisfaction (float): cluster size deviation from cluster average in percent  
        """
        
        x = X.to_numpy()

        closest_centroid_idx = np.ones(len(x)) * -1

        # Initial cluster centroids
        init_centroids = np.random.uniform(low=0, high=1, size=(self.k, self.n))

        for _ in range(max_it):
            cluster_means = np.zeros(shape=(self.k, self.n))

            # Find closest centroid
            for i, xi in enumerate(x):
                dists_to_centroids = [euclidean_distance(xi, centroid) for centroid in init_centroids]
                closest_centroid_idx[i] = dists_to_centroids.index(min(dists_to_centroids))

            # Calculate the cluster mean
            for ci in range(len(init_centroids)):
                cluster = np.empty(shape=(0, self.n))
                for i, xi in enumerate(x):
                    if ci == closest_centroid_idx[i]:
                        cluster = np.append(cluster, [xi],axis=0)
                if len(cluster) == 0:
                    cluster_means[ci] = np.random.uniform(low=0, high=1, size=self.n)
                else:
                    cluster_means[ci] = cluster.mean(axis=0)
            
            # Check for convergence
            if np.max(init_centroids - cluster_means) < threshold/100:
                
                _ , cluster_count = np.unique(closest_centroid_idx, return_counts=True)
                a = 100
                noise = np.random.randint(-a, a, size=(1,2)) / (a*a)

                # Check for satisfaction
                if np.min(cluster_count) < (len(x) / self.k)*(1-satisfaction) and np.linalg.norm(noise) != 0:

                    min_idx = np.where(cluster_count == np.min(cluster_count))[0]
                    max_idx = np.where(cluster_count == np.max(cluster_count))[0]

                    # If smallest found cluster is too small, reassign it close to the largest found cluster
                    cluster_means[min_idx] = cluster_means[max_idx] + noise
                    print("Cluster", min_idx, "with", np.min(cluster_count), "points got reassigned to", max_idx)

                else:
                    break

            # Assign the cluster means as the new centroids
            init_centroids = cluster_means

        self.centroids = init_centroids

    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        x = X.to_numpy()

        predictions = [-1] * len(x)

        for i, xi in enumerate(x):
            dists_to_centroids = [euclidean_distance(xi, centroid) for centroid in self.centroids]
            predictions[i] = dists_to_centroids.index(min(dists_to_centroids))
        
        return predictions


    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids
    
    
    
    
# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum() # HAD TO REMOVE 'axis=1' FROM SUM
        
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
  