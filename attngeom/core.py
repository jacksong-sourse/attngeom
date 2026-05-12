import numpy as np


class AttentionGeometry:
    def __init__(self, num_heads=8, embed_dim=None, normalize=True):
        if num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.normalize = normalize
        
    def _compute_scaled_dot_product(self, Q, K, V):
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.T) / np.sqrt(d_k)
        attention_weights = self._softmax(scores, axis=-1)
        output = np.matmul(attention_weights, V)
        return output, attention_weights
    
    def _softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    
    def _initialize_projections(self, input_dim):
        if self.embed_dim is None:
            self.embed_dim = input_dim
        
        if self.embed_dim < self.num_heads:
            self.embed_dim = self.num_heads
        
        if self.embed_dim % self.num_heads != 0:
            self.embed_dim = ((self.embed_dim // self.num_heads) + 1) * self.num_heads
        
        self.head_dim = self.embed_dim // self.num_heads
        
        scale = np.sqrt(2.0 / (input_dim + self.head_dim))
        
        self.W_q = np.random.randn(self.num_heads, input_dim, self.head_dim) * scale
        self.W_k = np.random.randn(self.num_heads, input_dim, self.head_dim) * scale
        self.W_v = np.random.randn(self.num_heads, input_dim, self.head_dim) * scale
        self.W_o = np.random.randn(self.num_heads, self.head_dim, self.embed_dim) * scale
    
    def _compute_pairwise_distances(self, X):
        n = X.shape[0]
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(X[i] - X[j])
        return distances
    
    def _compute_cosine_similarity(self, X):
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        return np.matmul(X_norm, X_norm.T)
    
    def _compute_affinity(self, X):
        cos_sim = self._compute_cosine_similarity(X)
        distances = self._compute_pairwise_distances(X)
        sigma = np.mean(distances) + 1e-8
        
        rbf_kernel = np.exp(-distances ** 2 / (2 * sigma ** 2))
        affinity = (cos_sim + rbf_kernel) / 2
        
        return affinity
    
    def fit(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if X.shape[0] == 0:
            raise ValueError("Input data cannot be empty")
        
        if X.shape[0] == 1:
            raise ValueError("Input data must have at least 2 samples")
        
        self.input_dim = X.shape[1]
        self._initialize_projections(self.input_dim)
        
        if self.normalize:
            X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        
        self.affinity_matrix = self._compute_affinity(X)
        self.data_mean = np.mean(X, axis=0)
        self.data_std = np.std(X, axis=0) + 1e-8
        
        return self
    
    def transform(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.normalize:
            X = (X - self.data_mean) / self.data_std
        
        n_samples = X.shape[0]
        
        head_outputs = []
        head_attentions = []
        
        for h in range(self.num_heads):
            Q = np.matmul(X, self.W_q[h])
            K = np.matmul(X, self.W_k[h])
            V = np.matmul(X, self.W_v[h])
            
            output, attention = self._compute_scaled_dot_product(Q, K, V)
            head_outputs.append(output)
            head_attentions.append(attention)
        
        head_outputs = np.array(head_outputs)
        head_attentions = np.array(head_attentions)
        
        attention_importance = np.mean(head_attentions, axis=(1, 2))
        attention_importance = self._softmax(attention_importance)
        
        weighted_outputs = []
        for h in range(self.num_heads):
            weighted = head_outputs[h] * attention_importance[h]
            weighted_outputs.append(np.matmul(weighted, self.W_o[h]))
        
        final_output = np.sum(weighted_outputs, axis=0)
        
        if self.normalize:
            final_output = (final_output - np.mean(final_output, axis=0)) / (np.std(final_output, axis=0) + 1e-8)
        
        return final_output
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def get_attention_weights(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.normalize:
            X = (X - self.data_mean) / self.data_std
        
        head_attentions = []
        
        for h in range(self.num_heads):
            Q = np.matmul(X, self.W_q[h])
            K = np.matmul(X, self.W_k[h])
            V = np.matmul(X, self.W_v[h])
            
            _, attention = self._compute_scaled_dot_product(Q, K, V)
            head_attentions.append(attention)
        
        return np.array(head_attentions)
    
    def get_geometry_features(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        attn_output = self.transform(X)
        affinity = self._compute_affinity(X)
        
        features = []
        for i in range(X.shape[0]):
            row_features = []
            row_features.extend(attn_output[i])
            row_features.extend(affinity[i])
            features.append(row_features)
        
        return np.array(features)


class DataGeometryAnalyzer:
    def __init__(self, num_heads=8, embed_dim=None):
        self.attention_geometry = AttentionGeometry(num_heads, embed_dim)
    
    def analyze(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        self.attention_geometry.fit(data)
        representation = self.attention_geometry.transform(data)
        attention_weights = self.attention_geometry.get_attention_weights(data)
        
        return {
            'representation': representation,
            'attention_weights': attention_weights,
            'num_heads': self.attention_geometry.num_heads,
            'embed_dim': self.attention_geometry.embed_dim
        }
    
    def extract_features(self, data):
        result = self.analyze(data)
        return result['representation']
