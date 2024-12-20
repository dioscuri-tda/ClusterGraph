# ClusterGraph Pruning Methods

Pruning is a process used to simplify the structure of a ClusterGraph by removing edges that do not significantly contribute to the geometric organization of the clusters. These methods aim to retain the most informative edges while removing redundant or unimportant ones.

## Pruning Methods Overview

ClusterGraph offers several pruning strategies, each targeting different aspects of the graph's structure:

1. **Threshold Pruning** – Remove edges based on a predefined distortion threshold.
2. **Iterative Greedy Pruning** – An iterative approach that removes edges while minimizing global metric distortion.
3. **Connectivity-based Pruning** – Removes edges based on their impact on the overall connectivity of the graph.

---

## 1. Threshold Pruning

### Description
Threshold pruning removes edges that cause a large metric distortion in the ClusterGraph. Each edge in the graph represents a distance between two clusters, and this pruning strategy eliminates edges with distortion above a specific threshold. Intuitively, edges with high distortion are considered "shortcuts" that do not reflect the true manifold structure of the data.

### Formula

The metric distortion for a path between two clusters \(i\) and \(j\) is given by:

$$
\delta_e = \frac{|C_i| + |C_j|}{\sum_{v \in V} \deg(v) |C_v|}
$$

Where \(C_i\) and \(C_j\) are the two clusters, and \(\deg(v)\) is the degree of vertex \(v\).


### Usage

The `threshold_pruning` method uses a predefined metric distortion threshold to prune the graph. If the distortion of an edge exceeds this threshold, the edge is removed.

```python
# Example of threshold pruning
metric_distortion_graph = cluster_g.prune_distortion(threshold=0.5)
```

In this example, any edge with a metric distortion greater than `0.5` will be removed from the graph.

### When to Use
- When you have a good idea of what constitutes an acceptable level of distortion.
- When you want a quick pruning method based on a simple metric.

### Pros
- Simple and easy to implement.
- Fast execution time.

### Cons
- May not always preserve the connectivity of important clusters if the threshold is set too aggressively.

---

## 2. Iterative Greedy Pruning

### Description
Iterative greedy pruning removes edges one at a time, focusing on minimizing the global metric distortion of the graph. The idea is to remove edges that lead to the least increase in overall distortion. The procedure is repeated iteratively, and at each step, the edge that causes the least increase in metric distortion is removed.

### Usage

The `prune_distortion()` method implements the iterative greedy pruning procedure. By default, this method uses the global metric distortion to decide which edges to remove.

```python
# Example of iterative greedy pruning
metric_distortion_graph, md = cluster_g.prune_distortion(knn_g=5, score=True)
```

Here, `knn_g=5` specifies the number of nearest neighbors to consider for each vertex. The `score=True` argument returns the evolution of the metric distortion over the course of the pruning process.

### When to Use
- When you want a more refined approach to pruning that balances metric distortion across the graph.
- When you want to visualize how the metric distortion evolves as edges are pruned.

### Pros
- More nuanced pruning compared to threshold pruning.
- Can retain meaningful edges while simplifying the graph.
- Provides insight into the impact of each pruning step.

### Cons
- May be computationally more expensive than threshold pruning.
- Requires multiple iterations to complete.

---

## 3. Connectivity-based Pruning

### Description
Connectivity-based pruning aims to maintain the overall connectivity of the graph while removing edges. This approach focuses on the quality of paths between vertices and removes edges that do not contribute to the best paths. The main goal is to preserve the connectivity of the graph while simplifying its structure.

### Path Quality Function

The quality of a path between two vertices \(i\) and \(j\) is defined as:

$$
q(P) = \sum_{\{i,j\} \in P} \frac{1}{d_\mathcal{C}(C_i, C_j)}
$$

where \(d_\mathcal{C}(C_i, C_j)\) is the distance between the clusters \(C_i\) and \(C_j\).

The connectivity between two vertices is defined as the maximum path quality of any path between them:

$$
\operatorname{conn}(i,j;E) = \max_{P \in \mathcal{P}(i, j)} q(P)
$$

The connectivity of the entire graph is the average connectivity over all vertex pairs.

### Usage

The `connectivity_pruning()` method removes edges that cause the most significant reduction in connectivity. The pruning process can be performed iteratively.

```python
# Example of connectivity-based pruning
pruned_graph = cluster_g.connectivity_pruning()
```

### When to Use
- When preserving the graph’s connectivity is crucial, and you want to avoid disconnecting the graph during pruning.
- When you have large graphs and want to ensure that the most important edges (in terms of connectivity) are retained.

### Pros
- Ensures that the graph remains connected after pruning.
- Focuses on maintaining the quality of paths between vertices.

### Cons
- Can be computationally expensive, especially for large graphs.
- May require multiple iterations to achieve optimal pruning.

---

## Choosing the Right Pruning Strategy

When selecting a pruning strategy, consider the following:

- **Threshold Pruning** is suitable for quick, simple pruning based on a predefined level of distortion.
- **Iterative Greedy Pruning** is ideal when you want to refine the graph iteratively, preserving important structures while simplifying the graph.
- **Connectivity-based Pruning** is the best choice when maintaining the graph’s connectivity is paramount, and you are willing to invest in a more computationally intensive approach.

---

## References

If you find these pruning strategies useful, please consider citing our work:

```bibtex
@misc{dłotko2024clustergraphnewtoolvisualization,
      title={ClusterGraph: a new tool for visualization and compression of multidimensional data}, 
      author={Paweł Dłotko and Davide Gurnari and Mathis Hallier and Anna Jurek-Loughrey},
      year={2024},
      eprint={2411.05443},
      archivePrefix={arXiv},
      primaryClass={cs.CG},
      url={https://arxiv.org/abs/2411.05443}, 
}
```