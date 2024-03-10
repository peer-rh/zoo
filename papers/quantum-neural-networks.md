---
aliases: 
tags:
  - quantum/ml
  - ai
  - "#paper/read"
4 Sentence Summary: "The authors propose a QNN architecture and training algorithms, which is suitable to NISQ-Era devices, since (i) it has low memory requirements (trade off with # of evaluations), (ii) generalises well and shows good performance, (iii) it is tolerant to noise in the data."
Citation: "K. Beer _et al._, “Training deep quantum neural networks,” _Nat Commun_, vol. 11, no. 1, p. 808, Feb. 2020, doi: [10.1038/s41467-020-14454-2](https://doi.org/10.1038/s41467-020-14454-2)."
---
- The Authors point out the need for novel machine learning approaches due to the decline of Moore's Law
	- **NOTE:** Currently ist seems like AI-Algorithms can scale very well horizontally, so the question arrises how relevant quantum neural networks are 
- The **Architecture** of the *QNN* is based on multiple learnable [[Unitary]] ops., where each layer has $U^{j}$ assigned to it
	- $\rho^{\text{out}}=\text{tr}_{\text{in,hidden}}(\mathcal{U}(\rho^{\text{in}}\otimes \ket{000\dots}\bra{000\dots}_{\text{hidden, out}})\mathcal{U}^{\dagger})$, where $\mathcal{U}:=U^{out}U^{L}\dots U^{1}$, see [[Trace]], [[quantum-physics]](density ops.)
	- Note that this can rewritten as $\rho_{\text{out}}=\mathcal{E}^{\text{out}}(\mathcal{E}^{L}(\dots\mathcal{E}^{1}(\rho^{\text{in}})))$, where $\mathcal{E}^{l}(X^{l-1})=\text{tr}\left( \prod^{1}_{j=m_{l}}U^{l}_{j} (X^{L-1}\otimes \ket{000}\bra{000}_{l})\prod_{j=1}^{m_{l}}U^{l\dagger}_{j}\right)$, which highlights the *feed-forward* nature of the network
- The **cost function** is the *fidelity* and defined as $C=\frac{1}{N}\sum^{N}_{x=1}\braket{ \phi^{\text{out}}_{x} |\rho^{\text{out}}_{x}|\phi^{\text{out}}_{x}  }$, where $C=1$ is best and $C=0$ is worst
	- This cost function only works with pure states, and otherwise we must use *fidelity for mixed states*
	- This function is a good candidate, because (i) it is easy to simulate and (ii) it is a generalisation of the classical risk function
- The **optimisation algorithm** is $U\mapsto e^{i\epsilon K}U$, where $\epsilon$ is the learning rate, and $K$ is chosen to increase $C$ the most (See paper for full formula in Box 1)
- Note that $K$ only depends on this and the next layer $\implies$ we only need to store 2 layers at a time $\implies$ This is a good candidate for current NISQ architectures (fairly high computation speed, low memory)
	- Note that we must have input data, which is easily reproducible/copyable for this to work since we require multiple copies during training
- The authors carried out 2 **experiments** on a q. computer simulation
	- It showed promising results on training on random data (similar to theoretically best Unitary)
	- It showed that it isn't vulnerable to corrupted training data (decoherence)