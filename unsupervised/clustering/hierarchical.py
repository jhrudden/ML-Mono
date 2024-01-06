import numpy as np
from queue import PriorityQueue

class Hierarchical:
    def __init__(self, linkage='centroid'):
        if linkage not in ['centroid']:
            # TODO: add ward, single and complete linkage
            raise ValueError('Invalid linkage method')
        self.linkage = linkage
        self.clusters = None
        self.X = None

    def _calc_initial_distances(self):
        """
        Initialize the distance matrix between all pairs of samples (clusters)
        This follows the agglomerative approach, where each sample is initially a cluster
        
        Returns:
            distances: (n_samples, (dist, cluster_i, cluster_j)) priority queue
        """
        queue = PriorityQueue()
        for i in range(self.X.shape[0]):
            for j in range(i+1, self.X.shape[0]):
                dist = self._calculate_distance([i], [j])
                queue.put((dist, i, j))
        return queue

    def _calculate_distance(self, cluster_i, cluster_j):
        """
        Calculate the distance between two clusters
        Params:
            cluster_i: indices of the samples in cluster i
            cluster_j: indices of the samples in cluster j
        Returns:
            dist: distance between the two clusters
        """
        if self.linkage == 'centroid':
            # calculate the centroid of each cluster
            centroid_i = np.mean(self.X[cluster_i], axis=0)
            centroid_j = np.mean(self.X[cluster_j], axis=0)
            dist = np.linalg.norm(centroid_i - centroid_j)
        return dist
    
    def _update_distances(self, distances, clusters, new_cluster, old_clusters):
        """
        Update the distance matrix after merging two clusters
        Params:
            distances: (n_samples, (dist, cluster_i, cluster_j)) priority queue
            clusters: dictionary of clusters
            new_cluster: index of the new cluster
            old_clusters: indices of the clusters that were merged
        """
        for i in clusters:
            if i in old_clusters:
                continue
            dist = self._calculate_distance(clusters[new_cluster], clusters[i])
            distances.put((dist, i, new_cluster))

    def _perform_clustering(self, k):
        """
        Perform the clustering
        Params:
            k: number of clusters
        """
        # Initialize the distance matrix
        distances = self._calc_initial_distances()

        clusters = {i: [i] for i in range(self.X.shape[0])}
        while len(clusters) > k:
            dist, i, j = distances.get()
            
            if i not in clusters or j not in clusters:
                continue
            
            new_cluster = clusters[i] + clusters[j]
            new_cluster_idx = len(clusters)
            clusters[new_cluster_idx] = new_cluster
            del clusters[i], clusters[j]

            self._update_distances(distances, clusters, new_cluster_idx, [i, j])
        
        return clusters
    
    def fit(self, X, k):
        """
        Fit the model to the data
        Params:
            X: (n_samples, n_features) array
            k: number of clusters
        """
        self.X = X
        un_normalized_clusters = self._perform_clustering(k)
        # side effect of _perform_clustering is that returned cluster indices are not in order / contiguous
        # need to reindex the clusters
        cluster_key_map = {i: j for j, i in enumerate(un_normalized_clusters)}
        self.clusters = {cluster_key_map[i]: [cluster_key_map[j] for j in un_normalized_clusters[i]] for i in un_normalized_clusters}

    
    def _calculate_distances_to_cluster(self, samples, cluster):
        """
        Calculate the distance between a matrix of samples and a cluster
        Params:
            samples: matrix of new samples
            cluster: indices of the samples in the cluster
        Returns:
            dists: distances between each sample in X_new and the cluster
        """
        if self.linkage == 'centroid':
            # calculate the centroid of the cluster
            centroid = np.mean(self.X[cluster], axis=0)

            # calculate the distance from each sample in X_new to the centroid
            dists = np.linalg.norm(X_new - centroid, axis=1)

            return dists
        else:
            raise UnimplementedError('Only centroid linkage is implemented')

    def predict(self, X_new):
        """
        Predict the cluster for each sample
        Params:
            X_new: (n_samples, n_features) array
        Returns:
            y_pred: (n_samples,) array
        """
        # need to calculate the distance between each sample and each cluster
        # create big matrix of distances
        dists = np.zeros((X_new.shape[0], len(self.clusters)))
        for i, cluster in enumerate(self.clusters):
            dists[:, i] = self._calculate_distances_to_cluster(X_new, cluster)
        
        return np.argmin(dists, axis=1)



        



        
        



