---
aliases: 
tags:
  - ai/method
  - paper/read
4 Sentence Summary: GaLore is a method to reduce memory footprint of optimisers, which can be utilised during pretraining and finetuning. It utilises a low-rank representation of gradients, in order to trade of some computational resources for memory. It performs better than low-rank parameter methods and almost as good as full-rank optimisers.
Citation: "J. Zhao, Z. Zhang, B. Chen, Z. Wang, A. Anandkumar, and Y. Tian, “GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection.” arXiv, Mar. 06, 2024. doi: [10.48550/arXiv.2403.03507](https://doi.org/10.48550/arXiv.2403.03507)."
---
- In order to reduce the memory needed to fine-tune people have often turned to **PEFT** (parameter efficient fine tuning)
	- **LoRA** is one of the most popular methods to do so and it works based on $W_{t}-W_{0}+AB$, where $W_{0}$ is a full rank initial matrix and $A,B$ are low-rank and arte the trainable parameters
- Methods, such as *ReLoRA* exist to adopt this principle to pretraining
- Other methods to reduce memory in pretraining include *subspace learning* and *AdaFactor* (similar to GaLore)
- All these methods however are not able to keep up with full-rank pretraining
	- This is because (i) low-rank matrices most probably aren't the optimal solution to the problem and (ii) potentially some changes in the gradient dynamics, due to the low rank of the gradient
- **GaLore** keeps the weight *matrices* full-rank, however reduces the *gradients* to low-rank
- The general update of parameters is $W_{t}=W_{t-1}+\eta \rho_{t}(G_{t-1})$, where $\rho_{t}$ is dependent on the *optimiser* and may have large memory requirements (as in *Adam*)
- The authors see that while, weights may not necessarily be low-rank, gradients most probably are to a certain degree (see paper for proof/derivation)
- Therefore **GaLore** (Gradient Low-Rank Projection) aims to only keep a "small core" of the optimiser states
## Algorithm
- $W_{t}=W_{t-1}+\eta \tilde{G}_{t-1}$, where $\tilde{G}_{t}=P_{t}\rho_{t}\left( P_{t}^{\intercal}G_{t}Q_{t} \right)Q^{\intercal}$, where $P \in\mathbb{R}^{m\times r},Q\in\mathbb{R}^{n\times r},r\ll m,n$ 
	- The authors utilise *Lipschitz-Continuity* to show that this actually converges (see paper) (*convergence* in [[numerical-methods-for-computer-science]])
- This also tells us, that to converge as fast as possible, $P,Q$ should reflect the *Eigenstates* corresponding to the highest *Eigenvalues* $\implies$ We use **SVD** of $G_{t}=USV^{\intercal}\approx \sum^{r}_{i=1}s_{i}u_{i}v_{i}^{\intercal}$, where $P_{t}=[u_{1},\dots,u_{r}]$ and $Q_{t}=[v_{1},\dots,v_{r}]$
	- Makes sense, because $U,V$ correspond to orthonormal Eigenbasis of $G^{\intercal}G$ and $GG^{\intercal}$
	- We choose the $i$ based on the highest singular values $\sigma$ in $S$
- If we keep $P,Q$ constant throughout the training process, we are limited to the subspace they produce $\implies$ We recompute them with a frequency of $T$
- To further reduce memory usage only $P$ or $Q$ is stored as results are comparable to using both matrices
- $\implies$ GaLore has a *Memory Requirement* of $mn+mr+2nr$ compared to $mn+3mr+3nr$ of *LoRA*
- The additional *hyper-parameters* introduced by GaLore are the *scaling factor* $\alpha$ (controls the strength of the low-rank update), $T$ and $r$
	- Authors found $T$ of up too $1000$ to work well
	- As for $r$, higher ranks perform better (tradeoff is close to linear), however a rank of $128$ for 80k steps performed better than a rank of $512$ for 20k steps
## Results
- GaLore was combined with *$8$-bit optimisation* and *per-layer weight updates* to maximally reduce memory usage
- For Pretraining it greatly outperformed all low-rank methods and performed similar to $8$-bit Adam
	- It also scaled well to *Llama-7B* showing that it can also scale to larger architectures
- The overhead of throughput compared to regular $8$-bit Adam is around $17\%$ 
	- Implementation isn't fully optimised yet, so this will probably go down with more research