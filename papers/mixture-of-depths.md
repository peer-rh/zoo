---
aliases: 
tags:
  - ai/method/llm
  - ai/problem/attention
4 Sentence Summary: This paper can be thought of as a modified version of Mixture of Experts, which is based on the notion that not all tokens require the same amount of compute. This one however doesn't route to some specific Expert MLP and instead decides whether a Attention Block should compute on a token or should simply be a residual connection to the next layer. This means that the network can put relatively little compute into most tokens
Citation: "Y. Sun et al., “Retentive Network: A Successor to Transformer for Large Language Models.” arXiv, Aug. 09, 2023. doi: 10.48550/arXiv.2307.08621. Available: http://arxiv.org/abs/2307.08621. [Accessed: Feb. 22, 2024]"
---
- Similarly to MoE models, the MoD model also use a *router* to decide whether to compute the attention step for a given token
- **Method**
	- (i) Define a **Compute Budget**: This determines how many tokens should be routed to each attention block - A vanilla [[transformer]] has a compute budget of 100%, and a compute budget of 20% means only 1 in 5 tokens will get computed in each block
	- (ii) Define a **Routing mechanism**: MoD simply routed between 2 options - (i) Attention and MLP or (ii) Residual Connection (identity)
	- (iii) Define **Routing Scheme**: There are 2 options presented by the authors - (i) **Token-choice** (A router chooses for every token one of many predefined paths) and (ii *used by paper*) **Expert-choice** (Each Block chooses top-k tokens)
		- (i) Can lead to load-balancing issues
		- (ii) Can lead to under/over computation of tokens
	- (iv) **Implementation**: To include the router weights along the gradient path, the router values get multiplied to the attention outputs
- Note that sampling the top-$k$ for each router is *non-causal*, however this can be fixed by simply creating a predictor, which predicts based on the router score, whether a token will be in top-k distribution
	- A simple MLP with a stop-gradient
	- 99% accuracy with fast convergence (empirical evidence)
- MoD can be combined with MoE $\implies$ **MoDE**
	- Staged MoDE implements first the MoD Router and than follows up with the MoE router
	- Integrated MoDE is like a MoE, but one of the Router paths is a residual connection
- **Results**
	- For same size transformers, the MoD performs better than the baseline transformer (efficiency and efficacy)
	- Can scale better (the optimal MoD has more params than a transformer)
	- Aggressive Capacity Reduction (12.5%) performed best
	- Both MoDEs outperform MoD and MoE baselines
- The authors suggest that such routing strategies may be useful when wanting to integrate other computationally expensive functions into neural nets
	- That way they can only be applied if necessary