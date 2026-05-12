"""
Example: Using attngeom for clustering analysis
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from attngeom import DataGeometryAnalyzer


def main():
    print("=== Clustering with attngeom ===")
    
    # Generate synthetic data with clusters
    np.random.seed(42)
    data, true_labels = make_blobs(
        n_samples=150,
        n_features=10,
        centers=5,
        cluster_std=0.8,
        random_state=42
    )
    print(f"Data shape: {data.shape}")
    print(f"True clusters: {np.unique(true_labels)}")
    
    # Step 1: Extract geometric features
    print("\nStep 1: Extracting geometric features...")
    analyzer = DataGeometryAnalyzer(num_heads=8, embed_dim=32)
    features = analyzer.extract_features(data)
    print(f"Features shape: {features.shape}")
    
    # Step 2: Cluster using KMeans
    print("\nStep 2: Clustering with KMeans...")
    kmeans = KMeans(n_clusters=5, random_state=42)
    predicted_labels = kmeans.fit_predict(features)
    
    # Step 3: Evaluate
    print("\nStep 3: Evaluating results...")
    silhouette = silhouette_score(features, predicted_labels)
    print(f"Silhouette Score: {silhouette:.4f}")
    
    # Step 4: Visualize
    print("\nStep 4: Visualizing...")
    
    # Reduce to 3D for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    vis_data = pca.fit_transform(features)
    
    fig = plt.figure(figsize=(12, 6))
    
    # True clusters
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(vis_data[:, 0], vis_data[:, 1], vis_data[:, 2], c=true_labels, cmap='viridis')
    ax1.set_title('True Clusters')
    
    # Predicted clusters
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(vis_data[:, 0], vis_data[:, 1], vis_data[:, 2], c=predicted_labels, cmap='viridis')
    ax2.set_title('Predicted Clusters (attngeom features)')
    
    plt.tight_layout()
    plt.savefig('clustering_result.png', dpi=150)
    print("Saved visualization to clustering_result.png")
    
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
