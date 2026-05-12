"""
Quickstart example for attngeom
"""

import numpy as np
from attngeom import AttentionGeometry, DataGeometryAnalyzer


def main():
    print("=== attngeom Quickstart ===")
    
    # Generate sample data
    np.random.seed(42)
    data = np.random.randn(100, 10)
    print(f"Input data shape: {data.shape}")
    
    # Example 1: Basic usage with AttentionGeometry
    print("\n1. Basic AttentionGeometry usage:")
    ag = AttentionGeometry(num_heads=8, embed_dim=64)
    representation = ag.fit_transform(data)
    print(f"Output representation shape: {representation.shape}")
    
    # Example 2: Get attention weights
    print("\n2. Get attention weights:")
    attention = ag.get_attention_weights(data)
    print(f"Attention weights shape: {attention.shape}")
    print(f"Attention weights sum (should be ~1): {np.sum(attention[0, 0]):.4f}")
    
    # Example 3: Using DataGeometryAnalyzer
    print("\n3. Using DataGeometryAnalyzer:")
    analyzer = DataGeometryAnalyzer(num_heads=8, embed_dim=64)
    result = analyzer.analyze(data)
    
    print(f"Representation shape: {result['representation'].shape}")
    print(f"Attention shape: {result['attention_weights'].shape}")
    print(f"Number of heads: {result['num_heads']}")
    print(f"Embedding dimension: {result['embed_dim']}")
    
    # Example 4: Extract features directly
    print("\n4. Extract features directly:")
    features = analyzer.extract_features(data)
    print(f"Features shape: {features.shape}")
    
    # Example 5: Geometry features
    print("\n5. Get geometry features:")
    geo_features = ag.get_geometry_features(data)
    print(f"Geometry features shape: {geo_features.shape}")
    
    print("\n=== Done ===")
    print("\nThe geometric representation preserves structural information")
    print("from the original data without requiring any training!")


if __name__ == "__main__":
    main()
