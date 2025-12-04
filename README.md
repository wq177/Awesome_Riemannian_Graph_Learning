# [A Survey on Riemannian Graph Learning: Towards Foundation Models](???)

There is a collection of state-of-the-art papers on Riemannian Graph Learning methods. We focus on geometric deep learning methods that study non-Euclidean manifolds and curvature-aware representations for graph learning, particularly in hyperbolic and spherical spaces.

:heartbeat: We welcome contributions and suggestions of interesting papers and implementations in the area of Riemannian geometry for graph learning.

:email: If you have any questions or find something missing, feel free to contact: lsun@bupt.edu.cn.

:star: If you find this repository helpful for your research or work, please consider giving it a star!

--------------

## What is Riemannian graph learning?

Riemannian Graph Learning is an emerging approach in graph representation learning that incorporates Riemannian geometry to model complex graph-structured data. Unlike traditional Euclidean graph learning methods, Riemannian Graph Learning embeds graphs in non-Euclidean spaces (e.g., hyperbolic or spherical spaces) to better capture hierarchical, scale-free, or other non-linear structures in networks.

## Important Survey Papers

| Year | Title                                                        |    Venue    |                            Paper                             | Code |
| ---- | ------------------------------------------------------------ | :---------: | :----------------------------------------------------------: | :--: |
| 2022 | **Hyperbolic Graph Neural Network：A Review of Methods and Applications** |    arxiv   | [Link](https://arxiv.org/abs/2202.13852) | - |

----

# Papers

## Hyperbolic manifold

| Year | Title | Venue | Code | Architecture | Paradigm |
|:----:|:---------------------------------------------------------------------------------------------------------------------------------------|:-----:|:----:|:---:|:---:|
| 2019 | **Continuous hierarchical representations with poincar´e variational auto-encoders** | NeurIPS | [Link](https://github.com/emilemathieu/pvae) | VAE | N/A |
| 2019 | **Hyperbolic Graph Convolutional Networks.** | NeurIPS | [Link](https://snap.stanford.edu/hgcn/) | Convolution | Semi-supervised |
| 2019 | **Hyperbolic Graph Neural Networks** | NeurIPS | [Link](https://github.com/facebookresearch/hgnn) | Convolution | Semi-supervised |
| 2020 | **H2KGAT: Hierarchical Hyperbolic Knowledge Graph Attention Network** | EMNLP | N/A | Convolution | Semi-supervised |
| 2020 | **Latent Variable Modelling with Hyperbolic Normalizing Flows** | ICML | N/A | Flow model and Flow matching | Generative |
| 2020 | **Low-Dimensional Hyperbolic Knowledge Graph Embeddings** | ACL | [Link](https://github.com/tensorflow/neural-structured-learning/tree/master/research/kg_hyp_emb) | Convolution | Semi-supervised |
| 2021 | **ACE-HGNN: Adaptive Curvature Exploration Hyperbolic Graph Neural Network** | ICDM | [Link](https://github.com/RingBDStack/ACE-HGNN) | Convolution | N/A |
| 2021 | **A Hyperbolic-to-Hyperbolic Graph Convolutional Network** | CVPR | N/A | Convolution | Semi-supervised |
| 2021 | **DataType-Aware Knowledge Graph Representation Learning in Hyperbolic Space** | CIKM | N/A | Convolution | N/A |
| 2021 | **HGCF: Hyperbolic Graph Convolution Networks for Collaborative Filtering** | WWW | [Link](https://github.com/layer6ai-labs/HGCF) | Convolution | Semi-supervised |
| 2021 | **Hyperbolic Variational Graph Neural Network for Modeling Dynamic Graphs** | AAAI | N/A | VAE | Generative |
| 2021 | **LEARNING HYPERBOLIC REPRESENTATIONS OF TOPO LOGICAL FEATURES** | ICLR | [Link](https://www.google.com/search?q=https://github.com/pkyriakis/permanifold/) | Convolution | Semi-supervised |
| 2021 | **Lorentzian Graph Convolutional Networks** | WWW | N/A | Convolution | Semi-supervised |
| 2022 | **Curvature Graph Generative Adversarial Networks** | WWW | [Link](https://github.com/RingBDStack/CurvGAN) | VAE | Generative |
| 2023 | **H-Diffu: Hyperbolic Representations for Information Diffusion Prediction** | TKDE | N/A | SDE | N/A |
| 2023 | **HGWaveNet: A Hyperbolic Graph Neural Network for Temporal Link Prediction** | WWW | [Link](https://github.com/TaiLvYuanLiang/HGWaveNet) | Convolution | Semi-supervised |
| 2023 | **Hyperbolic diffusion embedding and distance for hierarchical representation learning** | ICML | N/A | N/A | Unsupervised |
| 2023 | **Hyperbolic Geometric Graph Representation Learning for Hierarchy-imbalance Node Classification** | WWW | [Link](https://github.com/RingBDStack/HyperIMBA) | Convolution | Semi-supervised |
| 2023 | **Hyperbolic representation learning: Revisiting and advancing** | ICML | N/A | Convolution | Semi-supervised |
| 2023 | **Hyperbolic VAE via Latent Gaussian Distributions** | NeurIPS | [Link](https://github.com/ml-postech/GM-VAE) | VAE | N/A |
| 2023 | **Multi-Order Relations Hyperbolic Fusion for Heterogeneous Graphs** | CIKM | N/A | Convolution | Self-supervised |
| 2023 | **RotDiff: A Hyperbolic Rotation Representation Model for Information Diffusion Prediction.** | CIKM | [Link](https://github.com/PlaymakerQ/RotDiff) | Transformer | Generative |
| 2023 | **Self-supervised Continual Graph Learning in Adaptive Riemannian Spaces** | WWW | N/A | Convolution | Continual Learning |
| 2024 | **Enhancing Hyperbolic Knowledge Graph Embeddings via Lorentz Transformations** | ACL | [Link](https://www.google.com/search?q=https://github.com/LorentzKG/LorentzKG) | Convolution | Semi-supervised |
| 2024 | **Hypformer: Exploring Efficient Transformer Fully in Hyperbolic Space** | KDD | [Link](https://www.google.com/search?q=https://github.com/Graph-and-Geometric-Learning/hyperbolic-transformer) | Transformer | Semi-supervised |
| 2024 | **HypMix: Hyperbolic Representation Learning for Graphs with Mixed Hierarchical and Non-hierarchical Structures** | CIKM | N/A | Convolution | Unsupervised |
| 2024 | **Multi-Hyperbolic Space-Based Heterogeneous Graph Attention Network** | ICDM | N/A | Convolution | Semi-supervised |
| 2024 | **Residual Hyperbolic Graph Convolution Networks** | AAAI | N/A | Convolution | Semi-supervised |
| 2024 | **Toward a Manifold-Preserving Temporal Graph Network in Hyperbolic Space** | IJCAI | [Link](https://github.com/quanlv9211/HMPTGN) | Convolution | Semi-supervised |
| 2024 | **LSEnet: Lorentz Structural Entropy Neural Network for Deep Graph Clustering** | ICML | [Link](https://github.com/ZhenhHuang/LSEnet) | Clustering | Unsupervised |
| 2024 | **HGCH: A Hyperbolic Graph Convolution Network Model for Heterogeneous Collaborative Graph Recommendation** | CIKM | [Link](https://github.com/LukeZane118/HGCH) | Convolution | Semi-supervised |
| 2024 | **Hyperbolic Geometric Latent Diffusion Model for Graph Generatio** | NeurIPS | [Link](https://github.com/RingBDStack/HypDiff) | SDE | Generative |
| 2024 | **Spiking Graph Neural Network on Riemannian Manifolds** | NeurIPS | [Link](https://github.com/ZhenhHuang/MSG) | Spiking | N/A |
| 2025 | **THGNets: Constrained Temporal Hypergraphs and Graph Neural Networks in Hyperbolic Space for Information Diffusion Prediction** | AAAI | N/A | Convolution | Semi-supervised |
| 2025 | **HyperDefender: A Robust Framework for Hyperbolic GNNs** | AAAI | [Link](https://www.google.com/search?q=https://github.com/nikimal99/HyperDefender.git) | Convolution | Semi-supervised |
| 2025 | **Hyperbolic Graph Diffusion Model.** | AAAI | [Link](https://github.com/LF-WEN/HGDM) | SDE | Generative |
| 2025 | **MHR: A Multi-Modal Hyperbolic Representation Framework for Fake News Detection** | TKDE | N/A | Convolution | Semi-supervised |
| 2025 | **Hgformer: Hyperbolic Graph Transformer for Collaborative Filtering** | ICML | N/A | Transformer | Semi-supervised |
| 2025 | **Hyperbolic-PDE GNN: Spectral Graph Neural Networks in the Perspective of A System of Hyperbolic Partial Differential Equations** | ICML | [Link](https://github.com/YueAWu/Hyperbolic-GNN) | Convolution | N/A |
| 2025 | **Understanding and Mitigating Hyperbolic Dimensional Collapse in Graph Contrastive Learning** | KDD | N/A | Convolution | Unsupervised Learning |
| 2025 | **Towards Effective, Efficient and Unsupervised Social Event Detection in the Hyperbolic Space** | AAAI | [Link](https://github.com/XiaoyanWork/HyperSED) | Convolution | Semi-supervised |
| 2025 | **Hyperbolic Diffusion Recommender Model** | WWW | [Link](https://github.com/yuanmeng-cpu/HDRM) | SDE | N/A |
| 2025 | **Hyperbolic Multi-semantic Transition for Next POI Recommendation** | WWW | [Link](https://github.com/PlaymakerQ/HMST) | N/A | N/A |
| 2025 | **Hyperbolic Variational Graph Auto-Encoder for Next POI Recommendation** | WWW | N/A | VAE | N/A |
| 2025 | **Hyperbolic-Euclidean Deep Mutual Learning** | WWW | [Link](https://github.com/caohaifang123/H-EDML) | Convolution | Self-supervised  |
| 2025 | **VoRec: Enhancing Recommendation with Voronoi Diagram in Hyperbolic Space** | SIGIR | [Link](https://github.com/s35lay/VoRec) | Convolution | Semi-supervised  |
| 2025 | **Hyperbolic Multi-Criteria Rating Recommendation** | SIGIR | N/A | Convolution | Semi-supervised  |

## Spherical manifold 

| Year | Title | Venue | Code | Architecture | Paradigm |
|:----:|:---------------------------------------------------------------------------------------------------------------------------------------------------|:-----:|:----:|:---:|:---:|
| 2020 | **DeepSphere: a graph-based spherical CNN** | ICLR | [Link](https://github.com/deepsphere) | Convolution | Semi-supervised |
| 2022 | **Learning Hypersphere for Few-shot Anomaly Detection on Attributed Networks** | CIKM | N/A | Convolution | Semi-supervised |
| 2022 | **Spherical Message Passing for 3D Molecular Graphs** | ICLR | [Link](https://github.com/divelab/DIG) | Convolution | Semi-supervised |
| 2022 | **Spherical Graph Embedding for Item Retrieval in Recommendation System.** | CIKM | N/A | Convolution | N/A |

## Constant Curvature Space

| Year | Title | Venue | Code | Architecture | Paradigm |
|:----:|:---------------------------------------------------------------------------------------------------------------------------------------------------|:-----:|:----:|:---:|:---:|
| 2020 | **DyERNIE: Dynamic Evolution of Riemannian Manifold Embeddings for Temporal Knowledge Graph Completion** | EMNLP | [Link](https://github.com/malllabiisc/HyTE) | Convolution | Self-supervised |
| 2020 | **Mixed-curvature Variational Autoencoders** | ICLR | [Link](https://github.com/oskopek/mvae) | VAE | Unsupervised |
| 2021 | **Mixed-Curvature Multi-Relational Graph Neural Network for Knowledge Graph Completion** | WWW | N/A | Convolution | Semi-supervised |
| 2022 | **A Self-supervised Mixed-curvature Graph Neural Network** | AAAI | N/A | Convolution | Self-supervised |
| 2024 | **HGE: Embedding Temporal Knowledge Graphs in a Product Space of Heterogeneous Geometric Subspaces** | AAAI | [Link](https://github.com/NacyNiko/HGE) | Convolution | Self-supervised |
| 2024 | **IME: Integrating Multi-curvature Shared and Specific Embedding for Temporal Knowledge Graph Completion** | WWW | N/A | Convolution | Semi-supervised |
| 2025 | **Mixed-Curvature Multi-Modal Knowledge Graph Completion** | AAAI | N/A | Convolution | Semi-supervised |


## Product and Quotient Space

| Year | Title | Venue | Code | Architecture | Paradigm |
|:----:|:---------------------------------------------------------------------------------------------------------------------------------------------------|:-----:|:---:|:---:|:---:|
| 2020 | **Constant Curvature Graph Convolutional Network** | ICML | N/A | Convolution | Semi-supervised |
| 2020 | **DyERNIE: Dynamic Evolution of Riemannian Manifold Embeddings for Temporal Knowledge Graph Completion** | EMNLP | [Link](https://github.com/malllabiisc/HyTE) | Convolution | Self-supervised |
| 2020 | **Mixed-curvature Variational Autoencoders** | ICLR | [Link](https://github.com/oskopek/mvae) | VAE | N/A |
| 2021 | **Mixed-Curvature Multi-Relational Graph Neural Network for Knowledge Graph Completion** | WWW | N/A | Convolution | Self-supervised |
| 2022 | **A Self-supervised Mixed-curvature Graph Neural Network** | AAAI | N/A | Convolution | Self-supervised |
| 2024 | **HGE: Embedding Temporal Knowledge Graphs in a Product Space of Heterogeneous Geometric Subspaces** | AAAI | [Link](https://github.com/NacyNiko/HGE) | Convolution | Self-supervised |
| 2024 | **A Mixed-Curvature Graph Diffusion Model** | CIKM | [Link](https://github.com/ProGDM) | SDE | Generative |
| 2024 | **IME: Integrating Multi-curvature Shared and Specific Embedding for Temporal Knowledge Graph Completion** | WWW | N/A | Convolution | Semi-supervised |
| 2025 | **Toward a Unified Geometry Understanding: Riemannian Diffusion Framework for Graph Generation and Prediction** | NeurIPS | [Link](https://github.com/RingBDStack/GeoMancer) | SDE | Semi-supervised |
| 2025 | **GraphMoRE: Mitigating Topological Heterogeneity via Mixture of Riemannian Experts** | AAAI | [Link](https://github.com/RingBDStack/GraphMoRE) | Convolution | Self-supervised |
| 2025 | **Mixed-Curvature Multi-Modal Knowledge Graph Completion** | AAAI | N/A | Convolution | Unsupervised |
| 2025 | **RiemannGFM: Learning a Graph Foundation Model from Riemannian Geometry** | WWW | [Link](https://github.com/RiemannGraph/RiemannGFM) | GFM | Semi-supervised |

## Pseudo-Riemannian Manifold

| Year | Title | Venue | Code | Architecture | Paradigm |
|:----:|:-------------------------------------------------------------------------------------------------|:-----:|:----:|:---:|:---:|
| 2020 | **Ultrahyperbolic Representation Learning** | NeurIPS | N/A | Convolution | Semi-supervised |
| 2021 | **Ultrahyperbolic Neural Networks** | NeurIPS | N/A | Convolution | Semi-supervised |
| 2021 | **Directed Graph Embeddings in Pseudo-Riemannian Manifolds.** | ICML | N/A | Convolution | Semi-supervised |
| 2022 | **Pseudo-Riemannian Graph Convolutional Networks** | NeurIPS | [Link](https://www.google.com/search?q=https://github.com/xiongbo010/QGCN.) | Convolution | Semi-supervised |
| 2022 | **Ultrahyperbolic Knowledge Graph Embeddings** | KDD | N/A | Convolution | Semi-supervised |
| 2025 | **Pseudo-Riemannian Graph Transformer** | NeurIPS | [Link](https://github.com/xiongbo010/QGCN) | Transformer | Semi-supervised |

## Grassmann Manifold


| Year | Title | Venue | Code | Architecture | Paradigm |
|:----:|:-------------------------------------------------------------------------------------------------------------------|:-----:|:----:|:---:|:---:|
| 2022 | **Embedding graphs on Grassmann manifold** | Neural Network | [Link](https://github.com/conf20/Egg) | N/A | N/A |
| 2023 | **Building Neural Networks on Matrix Manifolds: A Gyrovector Space Approach** | ICML | [Link](https://github.com/zhiwu-huang/SPDNet) | Convolution | Semi-supervised |
| 2024 | **Matrix Manifold Neural Networks++** | ICLR | [Link](https://github.com/zhiwu-huang/SPDNet) | Convolution | Semi-supervised |
| 2024 | **Cross-View Approximation on Grassmann Manifold for Multiview Clustering** | TNNLS | N/A | N/A | Unsupervised |
| 2024 | **Multiple Kernel Clustering with Shifted Laplacian on Grassmann Manifold** | MM | N/A | N/A | Semi-supervised |


## SPD Manifold

| Year | Title | Venue | Code | Architecture | Paradigm |
|:----:|:-----------------------------------------------------------------------------------------------------------------------------------|:-----:|:----:|:---:|:---:|
| 2023 | **Building Neural Networks on Matrix Manifolds: A Gyrovector Space Approach** | ICML | [Link](https://github.com/zhiwu-huang/SPDNet) | Convolution | Semi-supervised |
| 2024 | **Matrix Manifold Neural Networks++** | ICLR | [Link](https://github.com/zhiwu-huang/SPDNet) | Convolution | Semi-supervised |
| 2024 | **Graph Neural Networks on SPD Manifolds for Motor Imagery Classification: A Perspective From the Time-Frequency Analysis** | TNNLS | [Link](https://github.com/GeometricBCI/Tensor-CSPNet-and-Graph-CSPNet) | Convolution | Semi-supervised |
| 2025 | **Learning to Normalize on the SPD Manifold under Bures-Wasserstein Geometry** | CVPR | [Link](https://github.com/jjscc/GBWBN) | N/A | N/A |


## Generic Manifold

| Year | Title | Venue | Code | Architecture | Paradigm |
|:----:|:---------------------------------------------------------------------------------------------------------------------------------------------------|:-----:|:----:|:---:|:---:|
| 2023 | **CONGREGATE: Contrastive Graph Clustering in Curvature Space** | IJCAI | [Link](https://github.com/CurvCluster/Congregate) | Clustering | Unsupervised |
| 2024 | **Motif-aware Riemannian Graph Neural Network with Generative-Contrastive Learning** | AAAI | [Link](https://github.com/RiemannGraph/MotifRGC) | Convolution | Self-supervised |
| 2024 | **R-ODE: Ricci Curvature Tells When You Will be Informed** | SIGIR | N/A | ODE | Generative |
| 2024 | **RicciNet: Deep Clustering via A Riemannian Generative Model** | WWW | N/A | Clustering | Unsupervised |
| 2024 | **Spiking Graph Neural Network on Riemannian Manifolds** | NeurIPS | [Link](https://www.google.com/search?q=https://github.com/ZhenhHuang/MSGM) | Convolution | Semi-supervised |
| 2025 | **CurvGAD: Leveraging Curvature for Enhanced Graph Anomaly Detection** | ICML | [Link](https://github.com/karish-grover/curvgad) | Convolution | Anomaly Detection |
| 2025 | **Pioneer: Physics-informed Riemannian Graph ODE for Entropy-increasing Dynamics** | AAAI | [Link](https://www.google.com/search?q=https://github.com/nakks2/Pioneer) | ODE | Semi-supervised |
| 2025 | **Robust Explanations of Graph Neural Networks via Graph Curvatures** | NeurIPS | [Link](https://github.com/yazhengliu/Robust_explanation_curvature) | N/A | N/A |
| 2025 | **Deeper with Riemannian Geometry: Overcoming Oversmoothing and Oversquashing for Graph Foundation Models** | NeurIPS | [Link](https://anonymous.4open.science/r/GBN-E854) | N/A | N/A |


----

# Coding

Below is a summary of several commonly used Python libraries for Riemannian graph learning.

## [Geomstats](https://github.com/geomstats/geomstats)

Geomstats is a Python package for computations, statistics, machine learning and deep learning on manifolds.

The package is organized into two main modules: geometry and learning. The module geometry implements differential geometry: manifolds, Lie groups, fiber bundles, shape spaces, information manifolds, Riemannian metrics, and more. The module learning implements statistics and learning algorithms for data on manifolds. Users can choose between backends: NumPy, Autograd or PyTorch.

#### Installation

```bash
pip3 install geomstats
```
or
```bash
conda install -c conda-forge geomstats
```

## [Geoopt](https://github.com/geoopt/geoopt)

A Python library designed for Riemannian optimization in PyTorch. It provides tools to work with Riemannian manifolds, which are used in geometric deep learning.

#### Installation

Make sure you have pytorch>=2.0.1 installed
There are two ways to install geoopt:

1. GitHub (preferred so far) due to active development

```bash
pip install git+https://github.com/geoopt/geoopt.git
```

2. pypi (this might be significantly behind master branch but kept as fresh as possible)
   
```bash
pip install geoopt
```

#### Key Features of Geoopt
Geoopt provides commonly used APIs for working with tensors, manifolds, optimizers, samplers, and more. For further information, please refer to the [official repository](https://geoopt.readthedocs.io/en/latest/).

## [GraphRicciCurvature](https://github.com/saibalmars/GraphRicciCurvature)

A python library to compute the graph Ricci curvature and Ricci flow on NetworkX graph. This work computes the Ollivier-Ricci Curvature, Ollivier-Ricci Flow, Forman-Ricci Curvature(or Forman curvature), and Ricci community detected by Ollivier-Ricci flow metric.
#### Package Requirement

* [NetworkX](https://github.com/networkx/networkx) >= 2.0 (Based Graph library)
* [NetworKit](https://github.com/kit-parco/networkit) >= 6.1 (Shortest path algorithm)
* [NumPy](https://github.com/numpy/numpy) (POT support)
* [POT](https://github.com/rflamary/POT) (For optimal transportation distance)
* [python-louvain](https://github.com/taynaud/python-louvain) (For faster modularity computation)

#### Installation

1. Installing via pip

```bash
pip3 install [--user] GraphRicciCurvature
```

- From version 0.4.0, NetworKit is required to compute shortest path for density distribution. If the installation of NetworKit failed, please refer to [NetworKit' Installation instructions](https://github.com/networkit/networkit#installation-instructions).

2. Upgrading via pip

To run with the latest code for the best performance, upgrade GraphRicciCurvature to the latest version with pip: 
```bash
pip3 install [--user] --upgrade GraphRicciCurvature
``` 

----

# Citation

If you find this review helpful and are interested in our work, please kindly cite our papers:

```
@article{sun2025riemanngfm,
  title={RiemannGFM: Learning a Graph Foundation Model from Riemannian Geometry},
  author={Sun, Li and Huang, Zhenhao and Zhou, Suyang and Wan, Qiqi and Peng, Hao and Yu, Philip},
  journal={arXiv preprint arXiv:2502.03251},
  year={2025}
}

@article{sun2024lsenet,
  title={Lsenet: Lorentz structural entropy neural network for deep graph clustering},
  author={Sun, Li and Huang, Zhenhao and Peng, Hao and Wang, Yujie and Liu, Chunyang and Yu, Philip S},
  journal={arXiv preprint arXiv:2405.11801},
  year={2024}
}

@inproceedings{sun2024riccinet,
  title={Riccinet: Deep clustering via a riemannian generative model},
  author={Sun, Li and Hu, Jingbin and Zhou, Suyang and Huang, Zhenhao and Ye, Junda and Peng, Hao and Yu, Zhengtao and Yu, Philip},
  booktitle={Proceedings of the ACM Web Conference 2024},
  pages={4071--4082},
  year={2024}
}

@article{sun2024spiking,
  title={Spiking Graph Neural Network on Riemannian Manifolds},
  author={Sun, Li and Huang, Zhenhao and Wan, Qiqi and Peng, Hao and Yu, Philip S},
  journal={arXiv preprint arXiv:2410.17941},
  year={2024}
}

@inproceedings{sun2024motif,
  title={Motif-aware riemannian graph neural network with generative-contrastive learning},
  author={Sun, Li and Huang, Zhenhao and Wang, Zixi and Wang, Feiyang and Peng, Hao and Yu, Philip S},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={8},
  pages={9044--9052},
  year={2024}
}

@inproceedings{sun2024r,
  title={R-ode: Ricci curvature tells when you will be informed},
  author={Sun, Li and Hu, Jingbin and Li, Mengjie and Peng, Hao},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={2594--2598},
  year={2024}
}

@inproceedings{wang2024mixed,
  title={A Mixed-Curvature Graph Diffusion Model},
  author={Wang, Yujie and Zhang, Shuo and Ye, Junda and Peng, Hao and Sun, Li},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  pages={2482--2492},
  year={2024}
}

@article{sun2024rcoco,
  title={Rcoco: contrastive collective link prediction across multiplex network in riemannian space},
  author={Sun, Li and Li, Mengjie and Yang, Yong and Li, Xiao and Liu, Lin and Zhang, Pengfei and Du, Haohua},
  journal={International Journal of Machine Learning and Cybernetics},
  volume={15},
  number={9},
  pages={3745--3767},
  year={2024},
  publisher={Springer}
}

@inproceedings{sun2023self,
  title={Self-supervised continual graph learning in adaptive riemannian spaces},
  author={Sun, Li and Ye, Junda and Peng, Hao and Wang, Feiyang and Yu, Philip S},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={37},
  number={4},
  pages={4633--4642},
  year={2023}
}

@inproceedings{sun2023congregate,
  title={CONGREGATE: Contrastive Graph Clustering in Curvature Spaces.},
  author={Sun, Li and Wang, Feiyang and Ye, Junda and Peng, Hao and Philip, S Yu},
  booktitle={IJCAI},
  pages={2296--2305},
  year={2023}
}

@inproceedings{sun2022self,
  title={A self-supervised riemannian gnn with time varying curvature for temporal graph learning},
  author={Sun, Li and Ye, Junda and Peng, Hao and Yu, Philip S},
  booktitle={Proceedings of the 31st ACM international conference on information \& knowledge management},
  pages={1827--1836},
  year={2022}
}

@inproceedings{sun2022self,
  title={A self-supervised mixed-curvature graph neural network},
  author={Sun, Li and Zhang, Zhongbao and Ye, Junda and Peng, Hao and Zhang, Jiawei and Su, Sen and Yu, Philip S},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={4},
  pages={4146--4155},
  year={2022}
}

@inproceedings{sun2021hyperbolic,
  title={Hyperbolic variational graph neural network for modeling dynamic graphs},
  author={Sun, Li and Zhang, Zhongbao and Zhang, Jiawei and Wang, Feiyang and Peng, Hao and Su, Sen and Yu, Philip S},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={5},
  pages={4375--4383},
  year={2021}
}

```
