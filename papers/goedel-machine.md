---
aliases: 
tags:
  - ai
4 Sentence Summary: 
Citation:
---
- The **Gödel Machine** is a *fully self-referential general problem solver* 
- The idea is that they can fully rewrite themselves, but only if a *proof searcher* (part of the initial algorithm) can proof the usefulness of the rewrite
- The setup of the computer is based on the *space-bound Turing machine*
	- Inside of the state encoded are the space, $time$, $x$ (Input Bit-string), $y$ (Output Bit-string) and $p(1)$ (the initial program)
- At any time $t$ we want to maximise $utility$ (future success)
	- A typical function of success is $u:\mathcal{S}\times \mathcal{E}\to \mathbb{R}$ ($\mathcal{S}$ is state, $\mathcal{E}$ is Environment) $u:=\sum^T_{\tau=time}$

