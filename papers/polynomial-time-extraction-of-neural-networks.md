---
aliases: 
tags:
  - ai
  - paper/read
4 Sentence Summary: 
Citation: "I. A. Canales-Martínez, J. Chavez-Saab, A. Hambitzer, F. Rodríguez-Henríquez, N. Satpute, and A. Shamir, “Polynomial Time Cryptanalytic Extraction of Neural Network Models.” 2023. Accessed: Mar. 15, 2024. [Online]. Available: [https://eprint.iacr.org/2023/1526](https://eprint.iacr.org/2023/1526)"
---
> **NOT FINISHED** (up to p. 12)
- In 2020 a method (**CJM20**) for parameter extraction of neural networks was shown, which required *polynomial* queries, but *exponential time*
	- This was due to the authors being unable to figure out the $+/-$ sign of the parameters $\implies$ exhaustive search
- This paper shows a technique, which efficiently finds these signs
- The paper assumes the following capabilities
	- (i) The network runs with arbitrarily high precision and the input can be modified with arbitrarily high precision
	- (ii) The network is a fully-connected ReLU DNN with no skip connection
	- 
- The technique works on direct output and softmax output and also on any activation function, which is piecewise linear
- As in CJM20 the method first computes the first layer, the second, ....
	- However here lays the first challenge since, except for the first layer, input must always be $>0$ since ReLU
- *Peeling of a layer* works in 2 steps: (i) get some multiplicative of each neuron's weights and bias, and (ii) getting the *sign of the neuron* (multiplicative factor)
	- **NOTE:** While positive signs are a symmetry in the network, negative signs are not
- Step (i) is done the same as in CJM20
- For step (ii) this is novel approaches introduced by the paper and we can choose one of three methods (SOE, Neuron Wiggle, Last Hidden Layer)
### CJM20
- Is based on the linear function between $x_{1}, x_{2}$, $\mu: [0,1]\to \mathbb{R}^{n}$, where $f(\mu)$ is a *piecewise linear function* (based on piecewise linear activation)
- 
### SOE
- **SOE** (System of Equation) - Limited by the fact that the layer has to be very *contractive*, however very efficient as it is only one SOE per layer (compared to one per neuron in *Freeze* in CJM20)
### Neuron Wiggle
- **Neuron Wiggle** is for when the layer is not contractive enough (most often the case) and works on changing a value at some critical point slightly in two directions and observing the change in output
		- While the method in CJM20 requires *polynomial* oracle queries it requires *exponential* time, which this method does not
		- Since the *wiggle* may trigger wrong *signs*, this must be done at multiple independent inputs
		- In their experiments 200 datapoints were sufficient to compute the sign of a neuron
### Last Hidden Layer