---
aliases: 
tags:
  - "#ai/method/llm"
  - "#ai/problem/attention"
---
- The Idea is similar to the [[Beaconed Transformer]] Paper, which proposes to reduce past tokens into *beacons* where for example every $256$ tokens get reduced into $16$ beacons, which get treated just as tokens
- The Idea proposes to reduce the $\mathcal{O}(n^{2})$ for memory cost in [[GPT]] style networks, by reducing tokens, that are far away into a smaller space
	- The plan is to split up the tokens into "blocks", which are of a size, which is optimised for GPUs
	- Then we feed `[...past_hidden][past_block][this_block]` and outputs `[this_hidden][this_pred]`, so we reduce the past_block into the hidden state
- There are multiple challenges which have to be overcome
	1. Make the conversion from past_hidden, past_block to new hidden state efficient
	2. For short sequences we should not  have to use the entire hidden space
	3. We probably increase number of parameters without adding more expressiveness to the model $\implies$ Prob. Poorer Performance
- There are multiple options for this transformation 
	1. Convert every Block into $k$-beacons (like paper)
	2. Simply design some network (FFN, or Convolution) to create the hidden state
		- Conv. networks can be designed to not use full space when just getting started
		- However the transformations should be "index-dependent"
			- Maybe one can review capsule networks and see if this is something worth investigating