---
aliases: 
tags:
  - "#quantum/ml"
  - "#paper/read"
4 Sentence Summary: This paper gives a current overview of the current quantum machine learning landscape. For me personally this was mostly meant to give me a quick introduction and to give some inspiration for future research direction.
Citation: "K. Zaman, A. Marchisio, M. A. Hanif, and M. Shafique, “A Survey on Quantum Machine Learning: Current Trends, Challenges, Opportunities, and the Road Ahead.” arXiv, Oct. 16, 2023. Accessed: Feb. 22, 2024. [Online]. Available: [http://arxiv.org/abs/2310.10315](http://arxiv.org/abs/2310.10315)"
---
- The increase in compute demand and decrease in applicability of *Moore's Law* make q. computing a compelling solution
## Quantum Computing Primer
- [[quantum-physics]]
- **Correlation** states how much we can say about state $\ket{\psi }_{B}$ based on a measurement in system $\mathcal{H}_{A}$ (entanglement)
- **Quantum Supremacy** is the term describing q. computers ability to solve problems in feasible time, which classical computers may not be able to
	- The set of these problems is known as $BQP$ (bounded error, quantum, polynomial Time)
	- $P\subset BQP\subset NP$
	- This is usually achieved by a q. computer being able to represent all combination of $n$-bits in $n$-qubits, compared to $2^{n}$-bits
- We currently only have **Noisy Intermediate-Scale Quantum-Computers(NISQ)**, but the goal is to have **Fault Tolerant Quantum Systems (FTQS)**
## Quantum ML Algorithms
- A **Parametrised q. circuit(PQA)** is often used as the basis of most QML algorithms and is a q. circuit which depends on some free parameters
- A **Variational Quantum Algorithm(VQA)** combines classical optimisation with PQAs $\implies$ They can already be used on NISQ devices 
- A **Quantum Approximate Optimisation Algorithm(QAOA)** is used to approximate solutions for combinatorial problems and is a good candidate for q. supremacy in NISQ-era
- A **Variation Quantum Eigensolver(VQE)** uses both classical and q. computers to find the ground state of a physical system ($min_{\ket{\psi}}\langle \hat{H} \rangle$)
	- The algorithm consists of a q. trial state(*Ansatz*) and a classical optimisation algorithm
- **Quantum Annealing** solves combinatorial problems
	- This means we have to model our optimisation problem as a physical system (Ising Model)
	- It works on the principle of *Minimum Energy*
- [[quantum-neural-networks]] (**QNN**) are like classical neural networks, but applied to q. computers
	- There are also QCNNs, *quanvolutional NNs*(makes QCNNs more implementable), QRNNs, and QGANs
- **Quantum Kernel** map higher dim. features to higher dim. space
	- This can then be used for clustering, ...
## Quantum Data
- Quantum Data describes the states of a q. system, and/or the evolution of the q. system
- Usually this is done by providing influencing data, such as unitary ops., stationary states, ...
- Currently most real-world problems collect classical data (bits, real-valued)
- Currently the main research focus is in understanding how to **encode/embed** classical data into q. states (**state preparation**)
	- There are 4 main considerations: 
		- Must be representable of the data (bad representation $\implies$ bad model expressivity)
		- Nr. of Qubits must be efficient
		- Width of q. circuit to encode must be relatively small (Few parallel computations)
		- The q. state must be so, that future ops. (arithmetic, ...) can be applied to it
	- Some encoding methods include
		- **Basis Embedding** - Encodes real numbers in binary
		- **Amplitude Embedding** - Enables us to represent $n$-Real Numbers is $\log n$-qubits
		- **Angle Embeddings** - Apply *Pauli Rotation Gates* to the qubits
- Current *q. datasets* include **QDataSet**, **QM7/8/9**, **Tensorflow Quantum Data** (Encodes MNIST with one qubit per pixel), ...
## State of Quantum Computers
- Since the idea of *Feynman* in 1982 lots of development took place
- Currently q. computer can be classified as either q. gate circuits or q. annealers
- **Quantum Gate Circuits** apply a set of unitary operations on a system of qubits to achieve the desired result
	- Every q. gate circuit can be represented by $1$-qubit rotation gates and $2$-qubit $CNOT$-gates
	- The process of compiling the circuit on a a q. computer is known as **Q. Circuit Mapping**
- **Quantum Annealers** are discussed above
- **Quantum Simulators** are used to simulate q. computers on classical computers
	- This has the benefit that we can develop in noise-free development for algorithm research and can also add artificial noise if wanted