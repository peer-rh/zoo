---
aliases: 
tags:
  - ai
4 Sentence Summary: In this paper the authors give a rough overview of the current AI landscape as it comes to Mathematics. The main takeaway from the paper is not necessarily proposed methods, but much rather an actual definition of what makes a theorem useful/interesting and what is it that is missing from system 2 reasoning capabilities
Citation: "Y. Bengio and N. Malkin, “Machine learning and information theory concepts towards an AI Mathematician.” arXiv, Mar. 07, 2024. doi: [10.48550/arXiv.2403.04571](https://doi.org/10.48550/arXiv.2403.04571)."
---
- While current AI systems are performant in System 1 thinking they usually fail at system 2 level
	- System 1 is associated with *instincts* and *habitual*
	- System 2 requires *planning* and/or *reasoning*
- The authors aim to explore multiple ideas, on how to improve this with concepts from information theory and how these could be applied to an *AI Mathematician*
- One major difference that comes to mind between models, like [[transformer|Transformer Networks]] and humans is the *separation of thought/output loop* 
- Many argue that **Inductive Biases** (some structural constraints on a system causing the system to favour some solution) are one of the main reasons to intelligence 
- One inductive biases is the limited working memory the human brain has leading to the need of **compression**
	- This is the theory at the heart of *Occam's Razor*
	- Leading from this we can tell that we should prefer *theory* $t_{1}$ over $t_{2}$ if $t_{1}\equiv t_{2}$, but $|t_{1}|<|t_{2}|$ (**Fundamental Theory of Bayesian Inference**)
- Let $\mathcal{O}$ be the set of all valid model inputs (theorems) and $\mathcal{M}\subset \mathcal{O}$ the set of all physical inputs (provable theorems), then a *ML-Model* tries to predict, if $x \in \mathcal{M}$, based on $\mathcal{D}\subset \mathcal{M}$, even if $x$ is *outside* of sample $\mathcal{D}$
- The above statement on **generalisation** is merely focused on predicting if something is valid outside of the sample, but it does not focus on proofs
- If $S$ is the acyclic, directed Graph of all statements, then a *proof* would be path between two nodes
	- NOTE that the graph is *transitive* and if some path has been proven for $a,b$ the same path is applicable for $c,d$ if they share the same properties (think of subproofs)
- One of the main contributions of the paper is defining a *measure of usefulness of a theorem*
	- One is how well the set of $\{  t, T(\mathcal{O})\}$, where $t$ is the theorem in question, and $T(S)$ is the set of previous proofs, compresses $\mathcal{M}$, i.e how small is $\{ t, T(\mathcal{O}) \}$ and with $k$ steps how many statements in $\mathcal{M}$ can be derived
		- This of course can't be realistically computed, so the authors state that a function which can approximate this measure is helpful/necessary for advances in AI mathematicians
	- This of course doesn't account for factors such as *real-world usefulness* which has the been the main actor in human-research
- Another ingredient to an AI mathematician is the concept of *Active Learning*, i.e. the model takes some action and training data is based on that action $\implies$ **Exploration Problem**
	- Maybe the model can decide the usefulness of a theorem for itself, however then we must have some way to at least guide the model to find *interesting* theorems
- Another question that arrises is *generation* of theorems
	- Sampling and exhaustive search obviously are infeasible due to size of $\mathcal{O}$
	- Another option is **GFlowNet**, which can be trained on both positive samples and a reward function, which could efficiently sample theorems, which may be useful
- At the moment the best proof solvers are based on *Imitation Learning* (supervised-learning), however another option is **goal-conditioned reinforcement learning**
	- Both currently lack the essential part of setting subgoals, i.e. planning
- The best performing reinforcement algorithms usually rely on fairly broad tree searches, which are a lot more inefficient than humans, which usually only evaluate very few paths