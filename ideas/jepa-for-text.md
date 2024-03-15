---
aliases: 
tags:
  - "#ai/method"
---
- This idea is inspired by [[v-jepa]], but converts this to the domain of text
- We have encoder $E$ and predictor $P$ in standard JEPA and to make this "interfacable" we also include a decoder $D$
- The idea is that we encode text in latent space via $E$ $\to$ Develop thought/text in latent space via $P$ $\to$ Convert the latent-text to language via $D$
- This has the **benefit** that we could make latent tokens far bigger, reducing the memory need
	- For example we can make "the dog walked" "in the park" as two latent-encoded statements
- One **challenge** is finding a nice way to encode text in statements - One could use sentences or maybe some learnable tokeniser
- In comparison to *V-JEPA* this should probably have a rather small encoder/decoder and a larger predictor, since this is a lot more difficult and I have the feeling that encoding statements isn't that intensive
- Another **benefit** could be that the predictor can be used to explore the answer-space (tree of thought, ...) before returning an answer and converting back to text
- As for *training* a good strategy would be to either only train causal or too randomly mask out sentences 
	- To randomly mask out sentences/statements the model would have to be non-causal which is suboptimal
	- The problem is the information loss, since text is a lot more dense then video and not as repetitive
	- Maybe one could just mask out random characters with a null char and then just (i) fix the sentences and (ii) predict future tokens
- One downside is that this may very well not be good for storing exact data, such as numbers since the model may not learn well to reconstruct those
	- Perhaps a special dataset/loss function must be thought of, so like humans we give the model a more curricular dataset, with specialised cases
	- This may not be all to bad and [[LeCun]] et al. experienced the increase in *label efficiency* in JEPA-type models