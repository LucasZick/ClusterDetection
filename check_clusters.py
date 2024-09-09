#RUN THE FOLLOWING TO MAKE SURE YOU HAVE ALL THE DEPENDENCIES DOWNLOADED:
#   pip3 install -r requirements.txt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from graphic_stuff import GraphicStuff

# Function to load and clean the data
def load_and_clean_data(filepath):
    try:
        data = pd.read_csv(filepath, sep='\\s+', engine='python')
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.dropna()
        return data
    except Exception as e:
        print(f"Error loading and cleaning data: {e}")
        raise

# Function to preprocess the data
def preprocess_data(data):
    try:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        return data_scaled
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        raise

# Function to apply K-means clustering
def kmeans_clustering(data, n_clusters):
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data)
        return clusters, kmeans
    except Exception as e:
        print(f"Error applying K-means clustering: {e}")
        raise

# Function to apply Hierarchical Clustering
def hierarchical_clustering(data, n_clusters):
    try:
        Z = linkage(data, 'ward')
        clusters = fcluster(Z, n_clusters, criterion='maxclust')
        return clusters, Z
    except Exception as e:
        print(f"Error applying hierarchical clustering: {e}")
        raise

# Função para encontrar o melhor valor de eps usando o k-distância
def find_optimal_eps(data, min_samples):
    try:
        neigh = NearestNeighbors(n_neighbors=min_samples)
        nbrs = neigh.fit(data)
        distances, indices = nbrs.kneighbors(data)
        distances = np.sort(distances[:, min_samples - 1], axis=0)
        plt.figure('K-distance plot')
        plt.plot(distances)
        plt.title('K-distance plot for DBSCAN')
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'{min_samples}-th Nearest Neighbor Distance')
        plt.show()
        
        # Retornar o valor de eps sugerido (por exemplo, o ponto de cotovelo)
        # Aqui, pegamos o valor em torno do percentil 90, que muitas vezes é um bom ponto de inflexão
        return np.percentile(distances, 90)
    except Exception as e:
        print(f"Error in finding optimal eps: {e}")
        return 0.5  # Retorna um valor padrão se algo der errado

# Função DBSCAN ajustada
def dbscan_clustering_with_auto_eps(data, min_samples):
    try:
        # Calcula o eps sugerido automaticamente
        eps = find_optimal_eps(data, min_samples)
        print(f"Optimal eps calculated: {eps}")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(data)
        return clusters
    except Exception as e:
        print(f"Error applying DBSCAN clustering: {e}")
        raise

# Function to visualize clusters
def visualize_clusters(data, clusters, centers=None, title="Clusters"):
    try:
        plt.figure(title)
        unique_clusters = np.unique(clusters)
        for cluster in unique_clusters:
            plt.scatter(data[clusters == cluster, 0], data[clusters == cluster, 1], label=f'Cluster {cluster}', alpha=0.6)
        if centers is not None:
            plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=100, label='Cluster Centers')
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Error during visualization: {e}")

graphic_stuff = GraphicStuff()
selected_dataset = graphic_stuff.create_dataset_buttons()
graphic_stuff.show_loading_screen("Loading and Preprocessing Data...")

# Load and preprocess the data
try:
    data = load_and_clean_data(f'data/dataset{selected_dataset}.txt')
    processed_data = preprocess_data(data)
except Exception as e:
    print(f"Error during data loading or preprocessing: {e}")
    exit()

# K-means Clustering
n_clusters_range = range(1, 20)
distortions = []
silhouette_scores = []

try:
    for n_clusters in n_clusters_range:
        clusters, kmeans = kmeans_clustering(processed_data, n_clusters)
        distortions.append(kmeans.inertia_)
        if n_clusters > 1:
            score = silhouette_score(processed_data, clusters)
            silhouette_scores.append(score)
except Exception as e:
    print(f"Error during K-means clustering: {e}")
    exit()

graphic_stuff.hide_loading_screen()

try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),)
    
    # Plot the Elbow Method
    ax1.plot(n_clusters_range, distortions, 'bx-')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Distortion')
    ax1.set_title('Elbow Method')

    # Plot the Silhouette Score
    ax2.plot(n_clusters_range[1:], silhouette_scores, 'bx-')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score for K-means')

    plt.tight_layout()

    plt.show()
except Exception as e:
    print(f"Error during plotting: {e}")

# Hierarchical Clustering
try:
    graphic_stuff.show_loading_screen("Plotting Hierarchical Clustering Dendogram...")
    _, Z = hierarchical_clustering(processed_data, n_clusters)
    

    # Plot the dendrogram
    plt.figure('Hierarchical Clustering Dendrogram')
    dendrogram(Z)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    
    graphic_stuff.hide_loading_screen()
    plt.show()
    
except Exception as e:
    print(f"Error during hierarchical clustering or visualization: {e}")

# Apply DBSCAN
try:
    min_samples = 20  # Mantenha o valor de min_samples ajustável
    dbscan_clusters = dbscan_clustering_with_auto_eps(processed_data, min_samples)
    visualize_clusters(processed_data, dbscan_clusters, title="DBSCAN Clustering with Auto Eps")
except Exception as e:
    print(f"Error during DBSCAN clustering: {e}")

# Apply K-means with the ideal number of clusters
try:
    n_clusters = graphic_stuff.get_cluster_count()
    clusters, kmeans = kmeans_clustering(processed_data, n_clusters)

    centers = kmeans.cluster_centers_

    # Visualize the clusters and centers using the new function
    visualize_clusters(processed_data, clusters, centers=centers, title="Clusters and K-means Centers")


    graphic_stuff.show_centers(centers)
    
except Exception as e:
    print(f"Error during K-means clustering visualization or showing centers: {e}")
