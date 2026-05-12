# attngeom: Attention-based Geometric Representation

[![PyPI version](https://badge.fury.io/py/attngeom.svg)](https://badge.fury.io/py/attngeom)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/attngeom.svg)](https://pypi.org/project/attngeom/)

**颠覆传统数据表征方式，用Transformer多头注意力机制构建数据的几何表征。**

## 核心创新

传统数据表征方法存在信息失真问题，`attngeom` 提供了一种全新的思路：

- **无需预训练**：不套用完整Transformer，不进行任何训练
- **纯注意力分析**：使用多头注意力机制进行注意力分析后加权融合
- **几何视角**：从几何角度多维度构建数据表征
- **即插即用**：直接输入数据即可提取结构特征

## 原理

```
原始数据 ──► 多头注意力分析 ──► 加权融合 ──► 几何表征
```

1. **数据输入**：支持任意数值型数据
2. **多头注意力**：多个注意力头从不同角度分析数据关系
3. **加权融合**：根据各头重要性进行加权
4. **特征输出**：获得数据的几何表征

## 安装

```bash
pip install attngeom
```

## 快速开始

### 基本用法

```python
import numpy as np
from attngeom import AttentionGeometry

# 创建数据
data = np.random.randn(100, 10)  # 100个样本，每个10维

# 初始化注意力几何模型
ag = AttentionGeometry(num_heads=8, embed_dim=64)

# 拟合并转换数据
representation = ag.fit_transform(data)

# representation 即为数据的几何表征
print(representation.shape)  # (100, 64)
```

### 高级分析

```python
from attngeom import DataGeometryAnalyzer

# 创建分析器
analyzer = DataGeometryAnalyzer(num_heads=8, embed_dim=64)

# 分析数据
result = analyzer.analyze(data)

# 获取结果
representation = result['representation']    # 几何表征
attention_weights = result['attention_weights']  # 注意力权重

# 直接提取特征
features = analyzer.extract_features(data)
```

## 应用场景

- **数据特征提取**：为机器学习模型提供更有表达力的特征
- **数据可视化**：降维后进行可视化展示
- **异常检测**：发现数据中的异常模式
- **聚类分析**：基于几何表征进行聚类
- **推荐系统**：分析用户-物品交互的几何结构

## 核心优势

| 传统方法 | attngeom |
|---------|----------|
| 固定维度 | 自适应维度 |
| 单一视角 | 多头多视角 |
| 信息失真 | 保留结构信息 |
| 需训练 | 无需训练 |

## API 文档

### AttentionGeometry

```python
class AttentionGeometry(num_heads=8, embed_dim=None, normalize=True)
```

**参数**：
- `num_heads`: 注意力头数量，默认8
- `embed_dim`: 输出维度，默认等于输入维度
- `normalize`: 是否归一化数据，默认True

**方法**：
- `fit(X)`: 拟合数据，学习投影矩阵
- `transform(X)`: 转换数据为几何表征
- `fit_transform(X)`: 拟合并转换
- `get_attention_weights(X)`: 获取注意力权重
- `get_geometry_features(X)`: 获取几何特征

### DataGeometryAnalyzer

```python
class DataGeometryAnalyzer(num_heads=8, embed_dim=None)
```

**方法**：
- `analyze(data)`: 完整分析数据，返回包含表征和注意力权重的字典
- `extract_features(data)`: 直接提取特征

## 技术细节

### 注意力计算

采用Transformer的缩放点积注意力：

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

### 多头融合

每个注意力头独立计算，然后根据注意力重要性进行加权融合：

```
Output = sum(importance[h] * Head[h] @ W_o[h])
```

### 投影矩阵初始化

使用正交矩阵初始化投影矩阵，保证数值稳定性。

## 示例

### 聚类分析

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 提取特征
features = analyzer.extract_features(data)

# 聚类
kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(features)

# 评估
score = silhouette_score(features, labels)
print(f"Silhouette Score: {score}")
```

### 可视化

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 降维
pca = PCA(n_components=2)
visualization = pca.fit_transform(features)

# 可视化
plt.scatter(visualization[:, 0], visualization[:, 1], c=labels)
plt.show()
```

## 开发

```bash
# 克隆仓库
git clone https://github.com/jacksong-sourse/attngeom.git
cd attngeom

# 安装依赖
pip install -e .[dev]

# 运行测试
pytest tests/ -v
```

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 引用

如果您在研究中使用了`attngeom`，请引用：

```
@software{attngeom,
  title = {attngeom: Attention-based Geometric Representation},
  author = {宋梓铭},
  year = {2026},
  url = {https://github.com/jacksong-sourse/attngeom},
}
```

## 联系方式

- GitHub: [https://github.com/jacksong-sourse/attngeom](https://github.com/jacksong-sourse/attngeom)
- PyPI: [https://pypi.org/project/attngeom](https://pypi.org/project/attngeom)
- 邮件: 15011462616@163.com
