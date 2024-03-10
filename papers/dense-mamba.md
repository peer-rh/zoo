---
aliases: 
tags:
  - ai/method/llm
  - ai/problem/attention
  - paper/read
4 Sentence Summary: 
Citation: "W. He _et al._, “DenseMamba: State Space Models with Dense Hidden Connection for Efficient Large Language Models.” arXiv, Mar. 05, 2024. doi: [10.48550/arXiv.2403.00818](https://doi.org/10.48550/arXiv.2403.00818)."
---
- This paper tries to solve the problem of *hidden state* propagation from lower layers to higher-up layers in *State-Space* models, such as [[mamba]] and [[retentive-networks]]
- Clearly with deep nns the information of layer $l-m$ is hard to use in layer $l$, if $l\gg$ since the information underwent a lot of computation/modification, being passed through the network
- That's why the Authors propose the **DenseSSM** mechanism, which *fuses* the past $m$ layers states with the hidden state of layer $l$
- The mechanism is outlined by the general framework
	- $\mathcal{H}_{t}^{l}=[\phi(h_{t}^{l-1}),\dots,\phi(h^{l-m}_{t})]$, where $\phi$ is a *selection & projection* operator
	- ${h'}_{t}^{l}=\text{Fuse}(h^{l}_{t},\mathcal{H}_{t}^{l})$, where $\text{Fuse}$ fuses state and the selected state
- The authors chose $\phi(x)=\text{Linear(x)}\odot a_{t}^{l}$, where $a^{l}_{t}=\text{MLP}(x)$ acts as a **gate** (*selection*) and uses *SiLU* and is $2$-layer and the $\text{Linear}$ acts as a **projection**
- The $\text{Fuse}(h, \mathcal{H}):=h+\sum_{i\in\mathcal{H}}i$, while other options would have been *concatenation* or *attention*
- These definitions are universal and fairly efficient. Additionally they can easily be used to also model in *convolutional* setting of *State-Space Models*
- For [[retentive-networks]] we apply these first to $k$ and $v$ independently and then use $k',v'$ to construct our State
## Results
- **DenseRetNet** greatly outperformed an equiv. RetNet and competed/outperformed a Llama-based equiv.
- **DenseMamba** was a little better then regular Mamba, however not quite as much as with RetNet
- As for the **projection** operation, the *identity* seems to strike a good balance between efficiency and efficacy
- A good selection of $m$ the authors show that performance increases for greater $m$, however $m=2$ seems to be the sweet spot for added parameters / efficacy