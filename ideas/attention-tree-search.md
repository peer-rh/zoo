---
aliases: 
tags:
  - "#ai/method/llm"
  - "#ai/problem/attention"
---
- The idea focuses on extending context length of the [[transformer]] by summarising multiple tokens into one key and searching that key if necessary
- This is done by summarising a block of $l$ tokens into $1$ token $k_{i}$ 
- If $k_{i}$ and $q_{n}$ cross some threshold we extend our attention to every token in block $i$
- Once we have followed the paths we wanted to take we can simply apply softmax to all the tokens
- **NOTE:** We need some way of summarising the keys without much data loss
	- The naive way would be summation/average of the $l$ into $k_{i}$ - [[retentive-networks]] suggest that this may work better then one would expect
		- However cancellation may be a problem
	- We use some function to maybe construct more than $1$ token
		- This could either be a neural net or some statistical analysis
		- We could for example generate $16$ essence tokens for a block of size $256$
		- This would be different to [[Beaconed Transformer]]/[[beaconed-transformer-idea]] in the sense that we still use the actual $kv$-pairs of the actual tokens for all relevant tokens
- To even further increase our context length we can make these tokenised blocks deeper-level
	- With simple depth-$1$ a block of $256$ tokens can "store" info for $~64k$ tokens, meanwhile a depth-$2$ block can "store" information for $16M$ tokens $\implies$ We don't have to have many branches in order to have access to a lot of information
- **NOTE:** This reduces computational need, but still demands the same storage requirement (However smaller attention maps) and the added branching may make the model not perform as well