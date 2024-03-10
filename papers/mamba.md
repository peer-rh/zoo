---
aliases: 
tags:
  - ai/method/llm
  - ai/problem/attention
  - paper/read
4 Sentence Summary: The Mamba architecture proposes a successor to the popular Transformer architecture. It is a type of State Space Model which utilises a Selection mechanism to add to the modelling capability. To overcome the performance benefit induced by the time-dependence, the authors engineered some custom kernels and designed the model around current hardware.
Citation: "A. Gu and T. Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces.” arXiv, Dec. 01, 2023. Accessed: Feb. 27, 2024. [Online]. Available: [http://arxiv.org/abs/2312.00752](http://arxiv.org/abs/2312.00752)"
---
- [[transformer|Transformer Networks]] currently dominate the space of frontier models, however they are inefficient (although effective) for *long sequences*
- **Mamba** tries to achieve the *modelling power* of transformers, while having the *scalability* of state space sequence models (SSMs)
- Current Structured State Space Sequence Models (S4s) operate as follows
	- We have parameters $(A,B,C,\Delta)$ and we compute $h_{t}=\bar{A}h_{t-1}+\bar{B}x_{t}$, $y_{t}=Ch_{t}$, where $\bar{A}, \bar{B}$ are *discretised* via some function based on $\Delta$
	- The Discretisation step adds some properties to $A,B$, such as *resolution invariance* and *ensuring model normalisation*
	- One can see, that this recurrent operation is equivalent to the convolution operation $\bar{K}=(C\bar{B}, C\bar{A}\bar{B}, C\bar{A}\bar{A}\bar{B}, \dots)$, $y=x\star \bar{K}$
- Some other approaches to replace Transformers include [[retentive-networks]], **RWKV**, **H3**, **Hyena**, **Linear Attention**

## Selection Mechanism
- Mamba wants to add a *Selection Mechanism*, by making the Parameters dependent on the input $\implies$ The model is less efficient, but more effective
	- In fact the authors argue that one main task/attribute of sequence models is their compression of context (Transformers - No Compression $\to$ Highly effective, Low efficiency; RNNs - Constant State $\to$ Low effectiveness, highly efficient)
	- They argue that efficiency and effectiveness are to a certain degree a question of tradeoff (one doesn't allow the other)
- Mamba makes $(B,C,\Delta)$ dependent on $x$ $\implies$ No longer *LIT* $\implies$ The convolution op. is no longer possible
	- $B,C=\text{Linear}(x)$ and $\Delta=\text{softplus}(\theta+\text{Linear}(x))$
- While this hurts performance, the authors use *SRAM* allocation, *kernel fusion* and a *parallel scan* algorithm to overcome the performance hit
- The *Intuition* behind the choice of $\Delta$ is based on a **gating mechanism**, i.e. It decides how much $x_{t}$ influences the hidden state $h_{t}$
- The Selection mechanism solves:
	- **Variable Spacing**: Helps filter through noise ("um")
	- **Filtering Context**: Many models can't properly utilise context, since they can't properly ignore irrelevant tokens

## Selection and MLP Fusion
- Instead of using the standard $\text{Attn}\to \text{MLP}$ block architecture as used in transformers, Mamba "gets rid" of the separate MLP block, by (i) expanding the block-input by factor $E$, (ii) perform the selective update, and (iii) scaling back down to input size $d$
	- **NOTE:** Most of the parameters are in the Linear Maps and not the *selection mechanism*

## Results
- Mamba outperforms all SSMs on the *selective copying task* and generalises the *Induction Heads* benchmark for length up to **4000x** what it was trained on (Other models achieve only 2x)
- In *Language Modelling* it outperforms every SSM and performs equivalent to Transformer++
- In *DNA Sequencing* it outperforms and out-scales Transformers
- As for efficiency, Mamba outperforms *FlashAttention-v2* and scales particularly beneficial for (i) larger batch-size inference and (ii) long contexts