---
aliases: 
tags:
  - "#ai/method/llm"
  - "#implemented"
  - "#paper/read"
4 Sentence Summary: he Paper aims at solving the *"impossible Triangle"*, by achieving good performance and efficient inference and training. It does so by basically removing the non-linearity of softmax in the attention step and replacing it with a scaled-linearity, which means, that we can propagate a hidden state over time.
Citation: "Y. Sun _et al._, “Retentive Network: A Successor to Transformer for Large Language Models.” arXiv, Aug. 09, 2023. doi: [10.48550/arXiv.2307.08621](https://doi.org/10.48550/arXiv.2307.08621)."
---
- **Goal:** Solve the problem of *Good-Inference Time*, *Good Performance* and *Efficient Training*
	- [[transformer|Transformer Networks]] don't have Good Inference Time
- **Algorithm:** Similar to [[GPT]]'s, only that multi-head Attention is replaced with **Retention**
	- The Net has $L$ layers, with $h$ heads, and a dimensionality of $d$
	- Each layer consists of a **Gated Multi-scale Retention** Block followed by a FFN block ($Y=\text{gelu}(XW_{1})W_{2}$)
	- The **GMSR** Block works as follows$$\begin{align}
\text{head}_{i}&=\text{Retention}(X, \gamma_{i}) & \gamma_{i}=1-2^{-5-i} \\
Y&=\text{GroupNorm}([\text{head}_{0}, \dots])  \\
Y & =(\text{swish}(XW_{G})\odot Y)W_{O}
 & W_{G}, W_{O}\text{ are learnable}\end{align}$$
	- The **Retention Mechanism** is the main ingredient to the paper
	- We have $X_{n}, s_{n}\mapsto o_n$, where $s_{n}$ is a *hidden state* which is propagated over time, and we have learnable parameters $W_{Q}, W_{K},W_{V}$
	- We have $Q=\Theta(XW_{Q}), \ K=\bar{\Theta}(XW_{K}),\ V=XW_{V}$
	- The *recurrent* Algorithm is$$\begin{align} \Theta(x) & =e^{in \theta}x\\
s_{n} & =\gamma s_{n-1}+K_{n}^{\dagger}V_{n} \\ o_{n} & =Q_{n}S_{n} \end{align}$$
	- The *parallel* algorithm is $$\begin{align}
\Theta_{n} & =e^{in \theta} \\
D_{ij} & = \gamma^{i-j} \text{ if } i\geq j\text{ else } 0 \\
o & =(QK^{T}\odot D)V
\end{align}$$
		- *NOTE:* $D$ is both applies the $\gamma$-scaling and causal masking
	- *NOTE:* $\Theta$ can also be replaced with a real valued encoding method, such as [[XPos]]
	- The *chunk-wise* method can be constructed by combining the two methods
- **Implementation Details:** Since the group-norm in GMSR "scales out" scalar-multiplication we improve numerical stability by $QK^{T}\mapsto \frac{QK^{T}}{\sqrt{ d }}$ (very common), and $D_{ij}\mapsto \frac{D_{ij}}{\sqrt{ \sum_{k} D_{ik} }}$ 
- **Results:** Shows promising performance of PPL similar to, or better then Transformers for models of sizes > 2Billion params.
	- We have $\mathcal{O}(1)$ Inference cost and a $\mathcal{O}(n)$ Memory Requirement for long sequence prediction
	- The model is (obviously) a lot faster then a standard [[GPT]]-style Transformer (even with [[FlashAttention]])