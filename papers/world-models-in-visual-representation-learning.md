---
aliases: 
tags:
  - "#ai/method"
  - "#paper/read"
4 Sentence Summary: This paper explores utilising world models for self-supervised representation learning in images. It is based on the JEPA architecture and shows promising results in using the predictor network to fine-tune for tasks. It also offers guidelines on creating meaningful(equivariant) world models.
Citation: "Q. Garrido, M. Assran, N. Ballas, A. Bardes, L. Najman, and Y. LeCun, “Learning and Leveraging World Models in Visual Representation Learning.” arXiv, Mar. 01, 2024. Accessed: Mar. 04, 2024. [Online]. Available: [http://arxiv.org/abs/2403.00504](http://arxiv.org/abs/2403.00504)"
---
- Leveraging *world models* has been done to great extend in Reinforcement Learning, and this model explores their usage for visual representation learning
	- Before, while some supervised methods utilised world models for learning, after pre-training the world models usually gets discarded
- This paper introduces the **Image World Model (IWM)** which is based on I-[[JEPA]], by [[LeCun]] and adds *photometric transformations* to the *actions* of $P$
- They say a World Model is **invariant**, iff $P$ can not apply the transformations, otherwise it is **equivariant**
## The Model/Data
- The *Model* is the same as in IJEPA, and [[v-jepa]], so $P$ and $E$ are both Visual-[[transformer]]s, however $P$'s action also include the photometric transformations
- The *Loss* used is the same as in v-jepa, but in $\ell_{2}$, not $\ell_{1}$ (still with *exponential moving avg*)
- The *Data* is constructed via taking image $I$, then $y=F_{y}(I)$, where $F_{y}$ applies a *color jitter, random-flip, and random-crop* and $x=F_{x}(y)$, where $F_{x}$ applies *color jitter, masking, and **destructive-augmentation** (blur, grayscale)*
	- The action $a$ is simply $F^{-1}_{x}$ for some sample $x$
- The *evaluation* is done using the **mean reciprocal rank** (MRR), which takes some sample $I$ and construct $256$ variants of it, then it tries to convert the *latent* of the variant back to the *latent* of $I$ using $P$ and compares the results
	- A MRR $\approx 1$ means, that $P$ is good, while MRR $\approx 0$ means $P$ is bad
- To *condition* $a$, we have two options: 
	- (i) **Sequence Conditioning** - Adding tokens to represent the representation
	- (ii) **Feature Conditioning** (used by paper, since better down-stream performance)- Have a $d$-channel matrix of the transformations applied and use a $1\times1$ Convolutional Network to convert to the correct dim
- The paper uses *deep* Predictors, since these consistently led to more performant *IWM*s
	- Additionally it showed that *strong transformations* are important to create meaningful world models
## Results
- They show that this method outperforms other models (I-JEPA mainly, but also other pretrained *encoder-based* methods) on finetuning tasks 
	- More specifically it showed that *invariant* models are better for *encoder finetuning* and *equivariant* are better for *predictor, full-model finetuning* 
	- Additionally it showed that *predictor finetuning* and *full-model finetuning* are similar in performance, making finetuning $P$ very promising as it more efficient
