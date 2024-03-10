---
aliases: 
tags:
  - "#ai/governance"
  - "#paper/read"
4 Sentence Summary: The authors propose compute as a good regulation method, since it is *detectable, excludable, quantifiable and concentrated*. The authors provide some policy-examples which increase *visibility*, *allocate beneficially* and *enforce prohibitions*. However, the authors also point out some potential downsides regarding *privacy* and *future effectiveness* and guidelines to circumvent/compromise these.
Citation: "G. Sastry _et al._, “Computing Power and the Governance of Artificial Intelligence.” arXiv, Feb. 13, 2024. doi: [10.48550/arXiv.2402.08797](https://doi.org/10.48550/arXiv.2402.08797)."
---
### Why Compute?
- To create competent AIs one needs the *AI Triad* (Compute, Data, Algorithms/Human Expertise)
- The Authors propose that out of the 3 compute is the best to regulate, since
	- It has been the main actor in recent AI achievements (*Scaling Laws*, [[the-bitter-lesson]] states that upscaled general methods usually outperform specialised methods in the long-term, [[transformer]])
		- The amount of compute used for *frontier models* has doubled every 6 months since ~2010
	- It is **detectable** - It is a physical object, which cannot be cloned in comparison to data/algorithms (*copy problem*)
	- It is **excludable** - Governments can exclude bad actors from access to the technology
	- It is **quantifiable** - In comparison to data and algorithms there are clear measurements of compute (FLOP/s, Device communication speed, ...)
	- It is **concentrated** - The supply chain of High-End Chips is extremely concentrated
		- The Chip Manufacturing process is dominated by ASML, TSMC and Nvidia (Partly 100% Market Share in there areas of expertise)
			- Also extremely high barrier of entry
		- Large Scale Compute is mainly controlled by AWS, GCP and Azure which
		- 

### Execution
- To make policy-decision one needs **visibility** in AI development
	- Compute is a fairly accurate predictor of AI capabilities (at least as far as training objectives are concerned)
		- However it should not be used as the sole predictor of performance
	- Some proposed policies are (i) **required reporting of large-scale training**, (ii) **international AI-Chip registry**, (iii) **privacy-preserving workload monitoring**
		- (i) it is easy for the gov. to detect actors, which are capable of starting large-scale training making it easy to enforce
		- (ii) requires cooperation from many parties and there isn't really a good solution yet for cost-effective UIDs for chips
			- Add. this is a lot more involvement of the gov. in compute $\implies$ privacy concerns
		- (iii) Naive monitoring may be privacy intruding therefore one must use different measures such as inference/training signatures, encrypted logging of training details, which can be shown upon request or [[proof-of-learning]]
			- [ ] Proof of learning
- Governments can allow for more beneficial **allocation** of resources - 
	- Governments can prioritise certain technologies, such as defensive technologies and technologies for the public good
		- Can also halt the further development of some direction because of safety interests
	- Governments can allocate compute to non-private parties, such as civil-service and academia
	- Geopolitically governments and **positively/negatively redistribute** the compute to further advance objectives
- The authors explore the idea of a **CERN for AI** which would be a joint project from many geopolitical actors, which is for the public sector, i.e it would be used to improve frontier models, research AI safety and AI for public good (clean energy for example)
		-Since it is a non-private organisation it would lead to healthy competitiveness in the private sector
	- Also a democratic based model of the member states can lead to fair distribution of the model
	- The authors suggest that weights of the frontier model can either be distributed to licensed suppliers or via a API 
	- However the authors also point out that this might be very dangerous as (i) many people disagree on how such an  organisation should be structured and (ii) it may enable a bad actor to gain control over this centralised structure
	- 
- To **enforce** all these policies the authors suggest some methods
	- Creating a physical limit on device communication
	- Hardware-based remote enforcement
	- Multiparty control of risky training runs
	- digital norms being enforced by the IaaS providers (infrastructure as a service)

### Potential downsides
- Regulation of compute inherently goes with **privacy intrusion**, which may lead to more leakage-points for trade-secrets or other confidential data
- Regulating compute will probably also **worsen economic performance** (export control)
- **Centralised Control** of compute leads to a vulnerable system which bad-actors can take advantage of
- The effectiveness of compute regulations may worsen over time
	- *New Algorithms* may be developed which require less compute
	- *Narrow AIs* can be very dangerous and may not require large amounts of compute (AlphaFold, Un-training of Llama-2 Censorship for 200$)
	- *Cost of Compute* may drastically decrease making it more widely available
	- Other actors/governments will have *increased economic incentive to create their own chip-industry* $\implies$ Regulations will no longer be effective on them
- The being said the authors point out that most of these are not absolute and rather relative, so that *scaling* continues to pay dividends $\implies$ Only large investments can buy frontier AI
#### Potential Guidelines to Circumvent Downsides
1. Regulate only **large scale** AI training
	- Currently large-scale training runs can only be executed by a few actors, which makes this easy to enforce 
	- Currently all frontier models require large-scale training runs
2. Implement **privacy-preserving** practices
	- It must be made sure, that the policies do not intrude on consumer privacy and doesn't infringe on trade secrets
3. Focus compute-based controls where *ex-ante* measures are justified
	- *ex-ante* means before hand
	- Some things require regulations before they actually are done, since they may bear great risk to safety
4. **Periodically revisit** controls computing technologies
	- Since the field is rapidly evolving, the authors suggest that every policy should be revisited at-least once a year in order to assure that they aren't to loose/strict
5. Implement all controls with **substantive and procedural safe-guards**