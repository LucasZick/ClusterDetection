#RUN THE FOLLOWING TO MAKE SURE YOU HAVE ALL THE DEPENDENCIES DOWNLOADED:
#   pip3 install -r requirements.txt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
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

# Function to apply DBSCAN
def dbscan_clustering(data, eps, min_samples):
    try:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(data)
        return clusters
    except Exception as e:
        print(f"Error applying DBSCAN clustering: {e}")
        raise

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

# Apply K-means with the ideal number of clusters
try:
    n_clusters = graphic_stuff.get_cluster_count()
    clusters, kmeans = kmeans_clustering(processed_data, n_clusters)

    centers = kmeans.cluster_centers_

    # Visualize the clusters and centers
    plt.figure('Clusters and centers')
    for cluster in np.unique(clusters):
        plt.scatter(processed_data[clusters == cluster, 0], processed_data[clusters == cluster, 1], 
                    label=f'Cluster {cluster}', alpha=0.6)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=100, label='Cluster Centers')
    plt.title('Clusters and K-means Centers')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    graphic_stuff.show_centers(centers)
    
except Exception as e:
    print(f"Error during K-means clustering visualization or showing centers: {e}")
