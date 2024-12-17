---
aliases: 
tags:
  - ai/method/llm
date: 2024-06-16
---
- The theoretical basis of this method is a [[transformer]] where the $Q_{s},K_{s},V_{s}$ are not dense representations, but sparse features similar to the ones in [[scaling-monosemanticity]]
- Since $W_{Q_{s},K_{s},V_{s}}$ are very large this leads to (i) overparameterized setting *(good)* (ii) large memory and computational cost *(bad)*
- To circumvent this I propose a new method which works as follows
- We split $Q_{s}$ into multiple block $Q_{s}^i=W_{Q_{s}}^iX$ 
- Now the plan is to construct $Q_{d}^j=W_{Q_{d}}Q_{s}^{0\dots j}$, which is a dense representation of the block till $j$
	- We aim for $W_{R}(Q_d^{j})\approx Q_{s}^{0\dots j}$
- Now to train the Network we continually add new sparse blocks $\implies$ We will most likely need some structure to our training data
- We have $Q=\begin{bmatrix} Q_d^{j}\\Q_{s}^{j+1} \end{bmatrix}$, where new relations/features should be learned by the sparse $Q_{s}^{j+1}$ and existing features refined/strengthened with the $Q^{j}_{d}$
	- We use $\ell_{1}$ penalty on both the weights and features to encourage sparsity
- The learning rate for $Q_{s}$ is greater then for $Q_{d}$, effectively acting like gradient boosting
- After having trained on the batches as described below we initialise $Q_{s}^{j+2}$ and create $Q_{d}^{j+1}$
- We can repurpose the weights from $Q_{d}^{j}$ and just 'fine-tune' on new examples 
	- We also extend the dimension of $Q_{d}$ to include some more features
	- [ ] QUESTION: Is this even viable - Can we effectively add more sparse features to an already converged dense space
		- My hypothesis: Yes if we start with a large enough space. If not then this may be more complicated and will require smart addition of new dimensions and good order of data
## Modified Attention mechanism
- We do attention on $Q_{d}K_{d}V_{d}$, $Q_{S}K_{S}V_{S}$, and also additionally $Q_{s}K_{d}V_{d}$
- The first two are trivial (just standard attention), the third one is tricky
	- For sparse to dense conversion we use weights $C_{Q,K,V}$ $\implies$ We can simply use $(C_{Q}Q_{s})K_{d}$
- $out=\sigma(Q_{d}K_{d}^{T})V_{d} + \sigma(C_{Q}Q_{s}K_{d}^{T})V_{d} +\sigma(Q_{s}K_{s}^{t})(C_{V}V_{s})$ 
- [ ] How do we compute $C_{K}$
## Preparing the Data
- Since this method requires careful/good data order I believe a good approach would be the following
- We have one method in which we cluster the different texts into topic - Perhaps a smaller embedding model could be used
- We have one method with which we model the complexity of the text
	- Option (i): # of vocab and commonness of vocab
	- Option (ii: A ML model trained on a few labeled examples
- We then simply train new Sparse blocks on new topic clusters where over time we increase complexity.
	- In addition we randomly sample texts from already learned topics to keep old learned patterns
## Future Exploration
- After some training time the model can select/group training data by itself
---
# Redesign
We have previous layer activations $\begin{bmatrix} x^{(l-1)}_{D}\\x^{(l-1)}_{S} \end{bmatrix}$, weights $W_{\{Q,K,V\},\{D,S\}}$ and conversion weights $C_{Q,K,V}$. 
$$
\begin{align}
Q_{D}&=W_{QD}x^{(l-1)}_{D}\\
Q_{S}&=W_{QS}\begin{bmatrix} x^{(l-1)}_{D}\\x^{(l-1)}_{S} \end{bmatrix}
\end{align}
$$
$$\begin{align} 
x_{D}^{(l)}&=\sigma\left( Q_{D}K_{D}^{\intercal} \right)V_{D}+\sigma\left( C_{Q}Q_{S}K_{D}^{\intercal} \right)V_{D}
 \\
x_{S}^{(l)}&=\sigma\left( Q_{S}K_{S}^{\intercal} \right)V_{S}+\sigma\left( Q_{D}C_{K}K_{S}^{\intercal} \right)V_{S}
\end{align}
$$
- [ ] Do we want all of the cross references between sparse and dense blocks?
- 