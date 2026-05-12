import numpy as np
import pytest
from attngeom import AttentionGeometry, DataGeometryAnalyzer


class TestAttentionGeometry:
    def test_init_default(self):
        ag = AttentionGeometry()
        assert ag.num_heads == 8
        assert ag.embed_dim is None
        assert ag.normalize is True
    
    def test_init_custom(self):
        ag = AttentionGeometry(num_heads=4, embed_dim=32, normalize=False)
        assert ag.num_heads == 4
        assert ag.embed_dim == 32
        assert ag.normalize is False
    
    def test_fit_1d_data(self):
        data = np.random.randn(50)
        ag = AttentionGeometry()
        ag.fit(data)
        assert hasattr(ag, 'affinity_matrix')
        assert ag.affinity_matrix.shape == (50, 50)
    
    def test_fit_2d_data(self):
        data = np.random.randn(50, 10)
        ag = AttentionGeometry()
        ag.fit(data)
        assert hasattr(ag, 'affinity_matrix')
        assert ag.affinity_matrix.shape == (50, 50)
    
    def test_transform_shape(self):
        data = np.random.randn(50, 10)
        ag = AttentionGeometry(num_heads=8, embed_dim=32)
        representation = ag.fit_transform(data)
        assert representation.shape == (50, 32)
    
    def test_transform_1d_input(self):
        data = np.random.randn(50)
        ag = AttentionGeometry(num_heads=4, embed_dim=16)
        representation = ag.fit_transform(data)
        assert representation.shape == (50, 16)
    
    def test_get_attention_weights(self):
        data = np.random.randn(20, 5)
        ag = AttentionGeometry(num_heads=4, embed_dim=20)
        ag.fit(data)
        attention = ag.get_attention_weights(data)
        assert attention.shape == (4, 20, 20)
    
    def test_get_geometry_features(self):
        data = np.random.randn(20, 5)
        ag = AttentionGeometry(num_heads=4, embed_dim=20)
        ag.fit(data)
        features = ag.get_geometry_features(data)
        assert features.shape[0] == 20
        assert features.shape[1] == 20 + 20  # embed_dim + num_samples
    
    def test_attention_weights_sum(self):
        data = np.random.randn(20, 5)
        ag = AttentionGeometry(num_heads=4, embed_dim=20)
        ag.fit(data)
        attention = ag.get_attention_weights(data)
        
        for head in range(4):
            row_sums = np.sum(attention[head], axis=1)
            np.testing.assert_allclose(row_sums, np.ones(20), rtol=1e-5)
    
    def test_reproducibility(self):
        data = np.random.randn(30, 8)
        ag1 = AttentionGeometry(num_heads=4, embed_dim=16)
        ag2 = AttentionGeometry(num_heads=4, embed_dim=16)
        
        np.random.seed(42)
        rep1 = ag1.fit_transform(data)
        
        np.random.seed(42)
        rep2 = ag2.fit_transform(data)
        
        np.testing.assert_allclose(rep1, rep2)
    
    def test_empty_data(self):
        data = np.array([])
        ag = AttentionGeometry()
        with pytest.raises(Exception):
            ag.fit(data)
    
    def test_single_sample(self):
        data = np.random.randn(1, 10)
        ag = AttentionGeometry()
        with pytest.raises(Exception):
            ag.fit(data)
    
    def test_negative_num_heads(self):
        with pytest.raises(ValueError):
            AttentionGeometry(num_heads=-1)
    
    def test_zero_num_heads(self):
        with pytest.raises(ValueError):
            AttentionGeometry(num_heads=0)


class TestDataGeometryAnalyzer:
    def test_init(self):
        analyzer = DataGeometryAnalyzer(num_heads=4, embed_dim=32)
        assert analyzer.attention_geometry.num_heads == 4
        assert analyzer.attention_geometry.embed_dim == 32
    
    def test_analyze_output(self):
        data = np.random.randn(50, 10)
        analyzer = DataGeometryAnalyzer(num_heads=4, embed_dim=32)
        result = analyzer.analyze(data)
        
        assert 'representation' in result
        assert 'attention_weights' in result
        assert 'num_heads' in result
        assert 'embed_dim' in result
        
        assert result['representation'].shape == (50, 32)
        assert result['attention_weights'].shape == (4, 50, 50)
        assert result['num_heads'] == 4
        assert result['embed_dim'] == 32
    
    def test_extract_features(self):
        data = np.random.randn(50, 10)
        analyzer = DataGeometryAnalyzer(num_heads=4, embed_dim=32)
        features = analyzer.extract_features(data)
        assert features.shape == (50, 32)
    
    def test_analyze_list_input(self):
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        analyzer = DataGeometryAnalyzer(num_heads=2, embed_dim=6)
        result = analyzer.analyze(data)
        assert result['representation'].shape == (3, 6)


class TestEdgeCases:
    def test_identical_data(self):
        data = np.ones((20, 5))
        ag = AttentionGeometry(num_heads=1, embed_dim=5)
        representation = ag.fit_transform(data)
        assert representation.shape == (20, 5)
    
    def test_large_dimension(self):
        data = np.random.randn(30, 100)
        ag = AttentionGeometry(num_heads=8, embed_dim=64)
        representation = ag.fit_transform(data)
        assert representation.shape == (30, 64)
    
    def test_small_data(self):
        data = np.random.randn(10, 3)
        ag = AttentionGeometry(num_heads=2, embed_dim=6)
        representation = ag.fit_transform(data)
        assert representation.shape == (10, 6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
