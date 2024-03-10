---
aliases: 
tags:
  - "#ai/method"
  - "#paper/read"
4 Sentence Summary: The paper adapts LeCun's JEPA architecture to video and shows that it improves on current SOTA models or performs just as well. It shows that feature-learning is just as performant and has better label-efficiency/downstream task performance
Citation: "A. Bardes _et al._, “Revisiting Feature Prediction for Learning Visual Representations from Video | Research - AI at Meta,” _arxiv_, Feb. 2024, Accessed: Feb. 22, 2024. [Online]. Available: [https://ai.meta.com/research/publications/revisiting-feature-prediction-for-learning-visual-representations-from-video/](https://ai.meta.com/research/publications/revisiting-feature-prediction-for-learning-visual-representations-from-video/)"
---
- The paper tries to answer the question: *"How effective is feature prediction as a standalone objective for unsupervised learning from video with modern tools?"*
### Architecture
- The [[JEPA]] Architecture from [[LeCun]] uses two models, the *Encoder* $E_{\theta}$ (ViT - Visual [[transformer]]) and the *Predictor* $P_{\phi}$(Narrow [[transformer]]) where we try to map some input $x\mapsto y$ with a transformation $z$ 
- In this experiment $z=\Delta y$ which represents the spatio-temporal position of $y$
- The naive approach is $\text{argmin}_{\phi, \theta} \lVert P_{\phi}(E_{\theta}(x), \Delta y)- E_{\theta}(y) \rVert_{1}$ has the downside that constant $E_{\theta}$ is a trivial solution, leading to **representation collapse**
- $\implies$ We use $\text{argmin}_{\phi, \theta} \lVert P_{\phi}(E_{\theta}(x), \Delta y)- \text{sg}(\bar{E}_{\theta}(y) )\rVert_{1}$, where $\text{sg}$ is *stop-gradient* (no back-prop) and $\bar{E}$ is the *exponential-moving average* of $E_{\theta}$
	- The intuition behind this is that using $\bar{E}$ makes $P$ evolve faster than $E$ $\implies$ No feature collapse (lower chance at least; has been experimentally verified)
- Note that it uses $\ell_{1}$ and not $\ell_{2}$ loss, since the authors experience more stable performance with it
- The authors found that using a **learnable, non-linear** pooling function in $E$ resulted in far better performance compared to a linear function, such as $\text{avg}$ 
### Experiment
- The model *trains* on multiple (2M) videos and uses **multi-block** as a masking method
	- For a input video $V$, $y=\text{multi-block}(V)$ and $x=V\setminus y$ 
	- multi-block was found to be the highest performing masking strategy compared to *random-tube* and *causal-multi-block*
	- multi-block works by using a union of 8 random short-range masks and 2 long-range masks which results in masking roughly $90\%$ 
- *Evaluation* is done on action recognition, motion classification and action localisation for *videos* and object recognition, scene classification and fine-grained recognition for *images*
- *Results* are that this method outperforms or is up-to-par with SOTA models, which were trained using *pixel-prediction*
	- However it showed much better **label-efficiency** and **downstream task performance**
	- It is also shown that $P_{\phi}$ follows *spatio-temporal consistency*

