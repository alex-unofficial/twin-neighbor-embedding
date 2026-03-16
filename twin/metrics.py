import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances


def knn_classification_accuracy(X: np.ndarray, target: np.ndarray, n_neighbors):
    # Handle both scalar and iterable inputs
    if isinstance(n_neighbors, (list, np.ndarray)):
        accuracies = np.array([knn_classification_accuracy(X, target, n) for n in n_neighbors])

        return accuracies

    # Ensure X has shape (n_samples, d) matching target shape
    n_samples = len(target)
    if X.shape[0] != n_samples and X.shape[1] == n_samples:
        X = X.T

    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    knn_classifier.fit(X, target)

    # Generate target prediction from kNN classifier
    knn_prediction = knn_classifier.predict(None)

    # Compare predicted targets vs true targets to compute accuracy
    prediction_accuracy = np.mean(knn_prediction == target)

    return prediction_accuracy


def knn_neighbor_preservation_accuracy(X: np.ndarray, G: nx.Graph, n_neighbors):
    # Handle both scalar and iterable inputs
    if isinstance(n_neighbors, (list, np.ndarray)):
        accuracies = np.array([knn_neighbor_preservation_accuracy(X, G, n) for n in n_neighbors])

        return accuracies

    # Ensure X has shape (n_vertices, d) matching graph size
    n_vertices = G.number_of_nodes()
    if X.shape[0] != n_vertices and X.shape[1] == n_vertices:
        X = X.T

    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(X)

    # connectivity matrices (sparse)
    knn_adjacency_matrix = knn.kneighbors_graph(mode="connectivity")
    true_adjacency_matrix = nx.adjacency_matrix(G, weight=None)

    # ensure CSR format for elementwise operations
    knn_adj = knn_adjacency_matrix.tocsr()
    true_adj = true_adjacency_matrix.tocsr()

    # element-wise multiply to count matching neighbors
    matching_neighbors = knn_adj.multiply(true_adj)

    # row-wise sums -> 1D numpy arrays
    matching_counts = np.asarray(matching_neighbors.sum(axis=1)).ravel()
    true_degrees = np.asarray(true_adj.sum(axis=1)).ravel()

    # avoid division by zero: mark degree-0 nodes as NaN and use nanmean
    with np.errstate(divide="ignore", invalid="ignore"):
        neighbor_accuracy = matching_counts / true_degrees
    neighbor_accuracy = np.where(true_degrees == 0, np.nan, neighbor_accuracy)

    mean_accuracy = np.nanmean(neighbor_accuracy)

    return mean_accuracy


def point_distance_metric(X: np.ndarray, dim=2, return_histogram=False, bins="auto"):

    # Ensure X has shape (n_points, d)
    if X.shape[1] != dim and X.shape[1] == dim:
        X = X.T

    all_point_distances = euclidean_distances(X)

    if return_histogram:
        point_distance_histogram, bin_edges = np.histogram(all_point_distances, bins=bins)
        return point_distance_histogram, bin_edges
    else:
        average_point_distance = np.mean(all_point_distances)
        return average_point_distance
