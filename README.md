# GCLMI: Graph Contrastive Learning with Min-Max Mutual Information

This is a PyTorch implementation of GCLMI algorithm, which designs a new  graph contrastive learning framework to learn graph-level representations for both unsupervised and semi-supervised graph classification tasks. 



## Requirements

- python
- pytorch
- pytorch_geometric (pyg)

Note:

This code repository is built on [pyg](https://github.com/pyg-team/pytorch_geometric), which is a Python package built for easy implementation of graph neural network model family. Please refer [here](https://github.com/pyg-team/pytorch_geometric) for how to install and utilize the library.



### Datasets

Graph classification benchmarks are publicly available at [here](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).



### Run

To run GCLMI, just execute the following command for graph classification task:

```
python main.py
```

### Reference
[[1] Graph Contrastive Learning with Augmentations](https://github.com/Shen-Lab/GraphCL)
