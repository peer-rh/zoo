---
aliases: 
tags:
  - ai/method/llm
---
- The Idea is to modify the [[retentive-networks]] such that the decay is more expressive
- Currently $\gamma$ is assigned for each head and not-learnable
- The idea is to instead have $\alpha$, which is the decay for every dimension of $K$
- This can be easily implemented in recursive and chunk-wise (although one must limit themselves to a chunk-size < 64, because of numerical stability)
	- either $s_{n}=s_{n-1}\alpha +K^{T}V$, or $\text{attn}=\text{tril}((Q\alpha^{\text{0...csz}})(K\alpha^{0...-csz})^{T})$
- That being said it should be fairly easy to create a custom GPU kernel
- **Question:** Is the added expressiveness actually something worth it? Can the model construct something which it currently was unable to?
- **Challenge:** As of now the model doesn't implement positional encoding, in the hope that decay is enough to let the tokens reflect their position relative to each other. 
	- This is impossible in *selective copying*, since $k$ would have to be dependent on $h$ to make sure it doesn't overwrite a different $k$ and otherwise the order would not be kept (optionally $q$ would have to look up min. which it can't really do)
		- $\implies$ We have to either include pos. encoding, making the context length limited to the training
		- Or we have to find an efficient way to make $k$ dependent on $h_{t}$ removing the parallelism potentially