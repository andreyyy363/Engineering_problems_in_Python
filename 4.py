import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import AgglomerativeClustering, KMeans, MeanShift, DBSCAN, AffinityPropagation
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
from itertools import product
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set random seed for reproducibility
np.random.seed(42)


# 4.1 Load the Wine dataset without class labels
def load_data():
    # Use the same parameters as in Lab 3
    S1, S2, S3 = 5, 1, 1

    wine = load_wine()
    data, target = wine.data, wine.target

    # Select the data according to Lab 3 sampling
    class_0_indices = np.where(target == 0)[0][S1:S1 + 40]
    class_1_indices = np.where(target == 1)[0][S2:S2 + 40]
    class_2_indices = np.where(target == 2)[0][S3:S3 + 40]

    # Combine indices
    selected_indices = np.concatenate((class_0_indices, class_1_indices, class_2_indices))

    # Get features and labels
    X = data[selected_indices]
    y = target[selected_indices]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Return data, true labels, and feature names
    return X_scaled, y, wine.feature_names


# 4.2 Build dendrograms for all linkage methods
def build_dendrograms(X):
    methods = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']

    plt.figure(figsize=(20, 20))
    for i, method in enumerate(methods):
        plt.subplot(3, 3, i + 1)

        # Calculate the linkage matrix
        Z = linkage(X, method=method)

        # Plot the dendrogram
        dendrogram(Z)
        plt.title(f'Dendrogram using {method} linkage')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')

    plt.tight_layout()
    plt.savefig('dendrograms.png')
    plt.show()


# Calculate evaluation metrics
def evaluate_clustering(X, labels, true_labels):
    # For clusters with only one sample, silhouette score is not defined
    n_clusters = len(np.unique(labels))
    if n_clusters <= 1 or n_clusters >= len(X):
        silhouette = -1  # Invalid score
    else:
        try:
            silhouette = silhouette_score(X, labels)
        except:
            silhouette = -1

    # For other metrics, compare with true labels
    ari = adjusted_rand_score(true_labels, labels)
    nmi = normalized_mutual_info_score(true_labels, labels)

    return {
        'silhouette': silhouette,
        'ari': ari,
        'nmi': nmi,
        'n_clusters': n_clusters
    }


# 4.3.1 Agglomerative Clustering Analysis
def agglomerative_clustering_analysis(X, true_labels):
    n_clusters_range = range(2, 11)
    linkage_types = ['ward', 'complete', 'average', 'single']

    results = []

    for n_clusters, linkage_type in product(n_clusters_range, linkage_types):
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
        labels = model.fit_predict(X)

        metrics = evaluate_clustering(X, labels, true_labels)
        results.append({
            'n_clusters': n_clusters,
            'linkage': linkage_type,
            **metrics
        })

    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)

    # Visualize the results
    plt.figure(figsize=(18, 6))
    metrics_to_plot = ['silhouette', 'ari', 'nmi']

    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, 3, i + 1)

        for linkage_type in linkage_types:
            subset = results_df[results_df['linkage'] == linkage_type]
            plt.plot(subset['n_clusters'], subset[metric], 'o-', label=linkage_type)

        plt.xlabel('Number of clusters')
        plt.ylabel(f'{metric.upper()} score')
        plt.title(f'Agglomerative Clustering: {metric.upper()} vs. n_clusters')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('agglomerative_clustering.png')
    plt.show()

    # Find best parameters for each metric
    for metric in metrics_to_plot:
        best_idx = results_df[metric].idxmax()
        best_params = results_df.loc[best_idx]
        print(
            f"Best parameters for {metric}: n_clusters={best_params['n_clusters']}, linkage={best_params['linkage']}, score={best_params[metric]:.4f}")

    return results_df


# 4.3.2 K-means Clustering Analysis
def kmeans_clustering_analysis(X, true_labels):
    n_clusters_range = range(2, 11)
    results = []

    for n_clusters in n_clusters_range:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)

        metrics = evaluate_clustering(X, labels, true_labels)
        results.append({
            'n_clusters': n_clusters,
            **metrics
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Visualize the results
    plt.figure(figsize=(18, 6))
    metrics_to_plot = ['silhouette', 'ari', 'nmi']

    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, 3, i + 1)
        plt.plot(results_df['n_clusters'], results_df[metric], 'o-')
        plt.xlabel('Number of clusters')
        plt.ylabel(f'{metric.upper()} score')
        plt.title(f'K-means: {metric.upper()} vs. n_clusters')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('kmeans_clustering.png')
    plt.show()

    # Find best parameters for each metric
    for metric in metrics_to_plot:
        best_idx = results_df[metric].idxmax()
        best_params = results_df.loc[best_idx]
        print(f"Best parameters for {metric}: n_clusters={best_params['n_clusters']}, score={best_params[metric]:.4f}")

    return results_df


# 4.3.3 Mean Shift Clustering Analysis
def mean_shift_clustering_analysis(X, true_labels):
    # Determine bandwidth range
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(X)
    distances, _ = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, 4])  # 5th nearest neighbor

    # Create bandwidth range from data characteristics
    bandwidth_range = np.linspace(np.min(distances), np.max(distances), 10)

    results = []

    for bandwidth in bandwidth_range:
        model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        labels = model.fit_predict(X)

        metrics = evaluate_clustering(X, labels, true_labels)
        results.append({
            'bandwidth': bandwidth,
            **metrics
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Visualize the results
    plt.figure(figsize=(18, 6))
    metrics_to_plot = ['silhouette', 'ari', 'nmi']

    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, 3, i + 1)
        plt.plot(results_df['bandwidth'], results_df[metric], 'o-')
        plt.xlabel('Bandwidth')
        plt.ylabel(f'{metric.upper()} score')
        plt.title(f'Mean Shift: {metric.upper()} vs. bandwidth')
        plt.grid(True)

    # Plot number of clusters vs bandwidth
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['bandwidth'], results_df['n_clusters'], 'o-')
    plt.xlabel('Bandwidth')
    plt.ylabel('Number of clusters')
    plt.title('Mean Shift: Number of clusters vs. bandwidth')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('mean_shift_clustering.png')
    plt.show()

    # Find best parameters for each metric
    for metric in metrics_to_plot:
        best_idx = results_df[metric].idxmax()
        best_params = results_df.loc[best_idx]
        print(
            f"Best parameters for {metric}: bandwidth={best_params['bandwidth']:.4f}, n_clusters={best_params['n_clusters']}, score={best_params[metric]:.4f}")

    return results_df


# 4.3.4 DBSCAN Clustering Analysis
def dbscan_clustering_analysis(X, true_labels):
    # Determine eps range
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(X)
    distances, _ = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, 4])  # 5th nearest neighbor

    # Create eps range from data characteristics
    eps_range = np.linspace(np.min(distances), np.max(distances), 10)
    min_samples_range = [2, 3, 4, 5, 6]

    results = []

    for eps, min_samples in product(eps_range, min_samples_range):
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)

        # Handle noisy samples (labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Skip if no proper clusters found
        if n_clusters < 1:
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': 0,
                'silhouette': -1,
                'ari': -1,
                'nmi': -1
            })
            continue

        metrics = evaluate_clustering(X, labels, true_labels)
        results.append({
            'eps': eps,
            'min_samples': min_samples,
            **metrics
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Visualize the results
    plt.figure(figsize=(18, 6))
    metrics_to_plot = ['silhouette', 'ari', 'nmi']

    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, 3, i + 1)

        # Create pivot table for heatmap
        pivot = results_df.pivot_table(index='min_samples', columns='eps', values=metric)

        sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.2f')
        plt.title(f'DBSCAN: {metric.upper()} score')
        plt.xlabel('eps')
        plt.ylabel('min_samples')

    # Plot number of clusters in heatmap
    plt.figure(figsize=(8, 6))
    pivot_clusters = results_df.pivot_table(index='min_samples', columns='eps', values='n_clusters')
    sns.heatmap(pivot_clusters, annot=True, cmap='viridis', fmt='.1f')  # Изменено на '.1f'
    plt.title('DBSCAN: Number of clusters')
    plt.xlabel('eps')
    plt.ylabel('min_samples')

    plt.tight_layout()
    plt.savefig('dbscan_clustering.png')
    plt.show()

    # Find best parameters for each metric
    valid_results = results_df[results_df['n_clusters'] > 0]
    for metric in metrics_to_plot:
        if not valid_results.empty:
            best_idx = valid_results[metric].idxmax()
            best_params = valid_results.loc[best_idx]
            print(
                f"Best parameters for {metric}: eps={best_params['eps']:.4f}, min_samples={best_params['min_samples']}, n_clusters={best_params['n_clusters']}, score={best_params[metric]:.4f}")

    return results_df


# 4.3.5 Affinity Propagation Clustering Analysis
def affinity_propagation_analysis(X, true_labels):
    damping_range = np.linspace(0.5, 0.99, 10)
    results = []

    for damping in damping_range:
        try:
            model = AffinityPropagation(damping=damping, random_state=42, max_iter=300)
            labels = model.fit_predict(X)

            metrics = evaluate_clustering(X, labels, true_labels)
            results.append({
                'damping': damping,
                **metrics
            })
        except:
            results.append({
                'damping': damping,
                'n_clusters': 0,
                'silhouette': -1,
                'ari': -1,
                'nmi': -1
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Visualize the results
    plt.figure(figsize=(18, 6))
    metrics_to_plot = ['silhouette', 'ari', 'nmi']

    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(1, 3, i + 1)
        plt.plot(results_df['damping'], results_df[metric], 'o-')
        plt.xlabel('Damping')
        plt.ylabel(f'{metric.upper()} score')
        plt.title(f'Affinity Propagation: {metric.upper()} vs. damping')
        plt.grid(True)

    # Plot number of clusters vs damping
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['damping'], results_df['n_clusters'], 'o-')
    plt.xlabel('Damping')
    plt.ylabel('Number of clusters')
    plt.title('Affinity Propagation: Number of clusters vs. damping')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('affinity_propagation.png')
    plt.show()

    # Find best parameters for each metric
    valid_results = results_df[results_df['n_clusters'] > 0]
    for metric in metrics_to_plot:
        if not valid_results.empty:
            best_idx = valid_results[metric].idxmax()
            best_params = valid_results.loc[best_idx]
            print(
                f"Best parameters for {metric}: damping={best_params['damping']:.4f}, n_clusters={best_params['n_clusters']}, score={best_params[metric]:.4f}")

    return results_df


# 4.4 Visualize consolidated clustering results
def visualize_consolidated_results(all_results):
    plt.figure(figsize=(12, 8))

    metrics = ['silhouette', 'ari', 'nmi']
    algorithms = ['Agglomerative', 'K-means', 'Mean Shift', 'DBSCAN', 'Affinity Propagation']
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for metric in metrics:
        for i, (algo_results, algo_name) in enumerate(zip(all_results, algorithms)):
            # Filter out invalid entries
            valid_results = algo_results[algo_results[metric] > -1]

            if not valid_results.empty:
                plt.scatter(
                    valid_results['n_clusters'],
                    valid_results[metric],
                    label=f'{algo_name} - {metric.upper()}',
                    marker=markers[i],
                    color=colors[metrics.index(metric)],
                    alpha=0.7,
                    s=80
                )

    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Clustering Algorithms Performance Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('consolidated_results.png')
    plt.show()


# Load data
X, y, feature_names = load_data()
print("Data loaded successfully")
print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")

# Build dendrograms
print("\nBuilding dendrograms for all linkage methods...")
build_dendrograms(X)

# Run clustering analyses
print("\nRunning Agglomerative Clustering analysis...")
agglomerative_results = agglomerative_clustering_analysis(X, y)

print("\nRunning K-means Clustering analysis...")
kmeans_results = kmeans_clustering_analysis(X, y)

print("\nRunning Mean Shift Clustering analysis...")
mean_shift_results = mean_shift_clustering_analysis(X, y)

print("\nRunning DBSCAN Clustering analysis...")
dbscan_results = dbscan_clustering_analysis(X, y)

print("\nRunning Affinity Propagation Clustering analysis...")
affinity_results = affinity_propagation_analysis(X, y)

# Visualize consolidated results
print("\nVisualizing consolidated results...")
all_results = [
    agglomerative_results,
    kmeans_results,
    mean_shift_results,
    dbscan_results,
    affinity_results
]
visualize_consolidated_results(all_results)

print("Analysis completed successfully!")
