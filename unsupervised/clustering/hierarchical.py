import numpy as np
from queue import PriorityQueue

class Hierarchical:
    def __init__(self, linkage='centroid'):
        if linkage not in ['centroid', 'ward']:
            # TODO: add single and complete linkage
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
        elif self.linkage == 'ward':
            a_union_b = self.X[np.concatenate((cluster_i, cluster_j))]
            a = self.X[cluster_i]
            b = self.X[cluster_j]
            a_union_b_contrib = (a_union_b - np.mean(a_union_b, axis=0)) ** 2
            a_contrib = (a - np.mean(a, axis=0)) ** 2
            b_contrib = (b - np.mean(b, axis=0)) ** 2
            dist = np.sum(a_union_b_contrib) - np.sum(a_contrib) - np.sum(b_contrib)
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
            if i in old_clusters or i == new_cluster:
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
            new_cluster_idx = max(clusters.keys()) + 1
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
        self.clusters = {cluster_key_map[i]: un_normalized_clusters[i] for i in un_normalized_clusters}
    
    def fit_predict(self, X, k):
        """
        Fit the model to the data and predict the cluster labels
        Params:
            X: (n_samples, n_features) array
            k: number of clusters
        Returns:
            labels: (n_samples,) array of cluster labels
        """
        self.fit(X, k)
        labels = np.zeros(X.shape[0])
        for i, cluster in enumerate(self.clusters):
            sample_indices = self.clusters[cluster]
            labels[sample_indices] = i
        
        return labels
