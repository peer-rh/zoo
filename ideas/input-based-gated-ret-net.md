---
aliases: 
tags:
  - ai/method/llm
  - ai/problem/attention
---
- This is based on [[retentive-networks]], however instead of constant decay factors, this uses some $\alpha$ based on the input, so $h_{t}=(1-\alpha)h_{t-1}+\alpha (kv)_{t}$, similar to [[griffin]] and [[mamba]]
- Similar to this idea [[retnet-with-more-expressive-decay]] is another method, however this
	- Is much more compute-friendly (doesn't require unusual operations)
	- Is based on input
- The two concepts could also be combined, however this may be fairly redundant, since only scale back of $h$ would have impact ($\alpha k \equiv Ak$, where $A\in\mathbb{R}^{d}$)
- One downside to [[retentive-networks]] is that we would have to compute the decay every time again,
	- Additionally doing so is not trivial in an efficient manner
	- For RetNet we have $\sum _{k}\delta^{k} q_{t}k_{t-k}v_{t-k}$, however in our case we have $\sum_{k}\left( \prod_{i=k}^{n} \alpha_{i}\right)q_{t}k_{t-k}v_{t-k}$
	- To construct $\Delta$ for block-wise operations, we must 
		- (i) multiply $k$ with $\alpha$ and $q$ with $\alpha^{-1}$ (similar to impl. of [[retnet-with-more-expressive-decay]])
		- (ii) Construct the triangular matrix, $\Delta_{ij}=\prod_{k=i}^{j}\alpha_{k}$
- Further improvements could include
	- Local Attention as seen in [[griffin]]
	- Skip Connections as in [[dense-mamba]] (they also showed very strong improvement in ret net) when using these skip connections
