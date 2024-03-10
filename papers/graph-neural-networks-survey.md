---
aliases:
  - graph-neural-networks
tags:
  - "#ai/method"
  - "#paper/read"
4 Sentence Summary: The paper outlines the current standing in graph neural networks. It states that algorithms are highly dependent on graph and objective and currently there is no clear one-solution.
Citation: "J. Zhou _et al._, “Graph Neural Networks: A Review of Methods and Applications.” arXiv, Oct. 06, 2021. Accessed: Feb. 22, 2024. [Online]. Available: [http://arxiv.org/abs/1812.08434](http://arxiv.org/abs/1812.08434)"
---
- Machine Learning on Graphs has a long history - On the one side there is extending **deep learning** to *non-euclidean* geometries (Not a grid) and on the other there is **graph representation learning** where we try to extract features form our graph by reducing the dimensionality
- The general *Pipeline* for developing GNNs
	1. Find/Construct the **Graph** from our data
	2. Identify Graph **Type** and **Scale**
		- Type includes info on whether it is un-/directed, static/dynamic, homogeneous/heterogeneous, ...
		- A graph is identified as **large**, iff it doesn't fully fit on the current SOTA device, meaning it is dependent on the time of reference $\to$ We must employ some *sampling* methods
	3. Design a **loss-function** (Edge-Level, Node-Level, Graph-Level) (un-/semi-/supervised)
	4. Construct the Model using **Computation Modules**
- The authors highlight following **challenges** currently in GNNs:
	- **Label Efficiency** - There are some methods to improve this
	- **Robustness** - GNNs are (like NNs) vulnerable to Adversarial Attacks
	- **Interpretability** - It's not easy to reason about NNs/GNNs
	- **Pre-training** - Currently there isn't a lot of research on using large-scale unsupervised data to improve model performance
## Computational Modules
- There are 3 main types **Propagation Modules** (pass data between nodes), **Sampling Modules** (used for *large* graphs) and **Pooling Modules** (Extract data from multiple nodes)
- *Propagation Modules* can be divided into **spectral convolution**, **spatial convolution**, **attention based spatial operations**, **recurrent convergence operations**, **recurrent gated operations** and **skip-connections**
	- *Spectral Convolution* uses *Fourier Transform* to apply the convolution op (generally $g\star x=\mathcal{F}^{-1}(\mathcal{F}(g)\odot\mathcal{F}(x))$)
		- $g$ can be a learnable diagonal matrix, which makes this fairly efficient to compute
		- **Cheb-Net** uses *Chebychev-Polynomials* to approximate the computation
		- **GCN** (Graph Convolutional Network) only uses the *Chebychev-Polynomial* of degree $k=1$ to avoid *overfitting* and *exploding gradients*
		- **DGCN** (Dual GCN) uses two *GCNs* to learn *local* and *global consistency*
	- *Spatial Convolution* uses the graph directly and does not rely on converting the graph into *frequency-domain*
		- The **main challenge** lies in defining the conv operation for nodes with different size neighbourhoods
		- **Neural FPs** simply have different conv parameters for neighbourhoods of different sizes
		- **DCNN** (Diffusion Convolutional NN) uses *transition matrices* to represent the neighbourhoods of nodes
		- **PATCHY-SAN** extracts and normalises neighbourhoods of size exactly $k$ for every node in the graph and then computes on those
		- **LGCN** uses a *pooling operation* to limit the neighbourhood of a node to at most $k$ neighbours
	- *Attention Based Spatial Approaches* are based on the popular attention mechanism used in [[transformer]]
		- **GAT** (Graph Attention Network) adopts the attention method to attend each node to it's neighbourhood
	- Other *no-state* methods are **message-passing-networks**, **non-local NNs** and **"Graph Network"**
	- *Recurrent convergence-based methods* are similar to recurrent neural networks
		- **GNN** tries to learn a *state-embedding* of the neighbourhood and the node itself and then uses to hidden states to compute the output labels
		- **Graph-ESN** generalises the *Echo-State-Network* to graphs
		- **SSE** (Steady State Networks) consists of (i) computing the embeddings for each node and (ii) using the *steady state conditions* to map the embeddings on the *steady state constrained space*
	- *Recurrent gate-based methods* are similar to LSTMs and try to prevent/diminish *long-term memory loss* and methods include **GGNN**, and **GraphLSTM**
	- *Skip Connections* try to overcome the worse performance of *deep-NNs* and still habe the benefit of deep/far neighbourhoods
		- **Highway-GCN** uses *gating-weights* (known as *Highway Gates*)
		- **JKN** (Jump knowledge Network) learn *adaptive* and *structure-aware* representations of the graph
- Two common *Sampling Modules* are (i) only loading part of the neighbourhood randomly and (ii) limiting the neighbourhood of a node to **subgraphs**
- *Pooling Modules* usually are either **direct** (known from the nodes/vertices directly - *average, max, ...*) or **hierarchical** (also uses hierarchical structure of the whole graph to pool/condense information)

- In *unsupervised* setting one can use **Graph Auto-Encoders** to create a feature/latent space to represent the graph
	- These use a *GCN* to encode the Network and a simple decoder to convert it back to the adjacency matrix
	- There are also *variational* variant and **Adversarially Regularised Graph Auto encoders** (like GANs)