---
aliases:
  - hawk
tags:
  - ai/method
  - ai/problem/attention
  - "#paper/read"
4 Sentence Summary: The paper proposes a novel layer called RG-LRU, which is a recursive layer. Through custom kernels, they are able to match the speed of a Transformer. They show that a Hybrid of local attention and RG-LRU outperforms Transformers and other sequence models, while being very efficient to train and run inference on.
Citation: "S. De _et al._, “Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.” arXiv, Feb. 29, 2024. Accessed: Mar. 01, 2024. [Online]. Available: [http://arxiv.org/abs/2402.19427](http://arxiv.org/abs/2402.19427)"
---
- The paper proposes a novel layer (**RG-LRU**) to replace the attention mechanism used in [[transformer|Transformer Networks]]
- They show that it matches the efficiency of transformers and performs just as-well, if not better as [[mamba]]
- They train two types of model: (i) **Hawk**, which is only based on the RG-LRU and (ii) **Griffin** which is a hybrid of Local-Attention and RG-LRU
## Architecture
- Each model consist of $N$ **Residual Blocks** (standard encode/decode as in transformer around the sequence)
- The Residual Blocks work as follows
	- $x=x+\text{TemporalMixing}(\text{Norm}(x))$
	- $x=x+\text{MLP}(\text{Norm}(x))$
- The **MLP**-Block is a *gating-function* with an expansion factor $M$: $x=\text{Linear}_{C}(\text{GeLU(Linear}_{A}(x))\otimes \text{Linear}_{B}(x))$, where $A,B:\mathbb{R}^{d}\to \mathbb{R}^{Md}$ and $C: \mathbb{R}^{Md}\to \mathbb{R}^{d}$
- The **Temporal-Mixing**-Blocks come in 3 variants
	- *Global-Multi-Query-Attention* - Used as baseline comparison
	- *Local-Multi-Query-Attention* - Works the same as regular MQA, but is limited to past $k$ tokens
	- *RG-LRU* - The proposed recurrent block, which is inspired by *LSTMs* and *Linear-Recurrent-Units* and works as follows:
		- $r_{t}=\sigma(\text{Linear}_{r}(x))$ *(Recurrence Gate)* ($\sigma$ is *Sigmoid*)
		- $i_{t}=\sigma(\text{Linear}_{i}(x))$ *(Input Gate)*
		- $a_{t}=a^{8r_{t}}$, where $a=\sigma(\Lambda)$, $\Lambda$ is diagonal and uniform, so that $a\in[0.9, 0.999]$ 
		- $y_{t}=h_{t}=a_{t}\odot h_{t-1}+\sqrt{ 1-a^{2}_{t} }\odot(i_{t}\odot x_{t})$
		- One can see, that gates do not depend on $h_{t-1}$, but only on $x_{t}$ enabling fast computation (Similar as in [[mamba]])
		- $\implies$ This can also be interpreted as a $1D$-Convolutional Block, enabling faster performance and is bound by Memory-Transfer $\implies$ Custom kernel enables speedup
			- Even though it is possible to use a *associative-scan* the added memory transfers made this more costly than a *linear-scan*
- **NOTE:** Only relative position-encoding is used in the form of [[RoPE]] and no absolute position-embeddings
## Results
- They show that the performance of all models (MQA, Hawk, Griffin) scale linearly with training FLOP/s
- Hawk and Griffin have better *down-stream* performance than [[mamba]] and *Llama-2*, although having less training compute
- Hawk and Griffin training speed improvements only come for training on long-sequences (The quadratic time-complexity of transformers)
- In the *Selective Copying* and *Induction Heads* the 2 Models where able to extrapolate further than standard MQA
	- Hawk converged a lot slower than the other two models