---
aliases: 
tags:
  - ai
  - paper/read
4 Sentence Summary: 
Citation: "M. Li _et al._, “The Deep Learning Compiler: A Comprehensive Survey,” _IEEE Trans. Parallel Distrib. Syst._, vol. 32, no. 3, pp. 708–727, Mar. 2021, doi: [10.1109/TPDS.2020.3030548](https://doi.org/10.1109/TPDS.2020.3030548)."
---
- The paper gives a overview of the landscape of **Deep Learning compilers**
	- Note that the paper is already 3 years old, and most probably outdated (therefore I disregarded the benchmarks)
- AI *Hardware* can be separated into 3 groups
	- **General Purpose**: GPUs, CPUs
	- **Dedicated**: TPUs,...
	- **Neuromorphic**: Hardware inspired by the human brain
- The **frontend** of the the compiler is the framework used by the programmer
	- **Node-Level** optimisations replace/remove specific nodes. They include *Nop-opt.* (No Operations) and *zero-dim-opt.*
	- **Block-Level** optimisations include (i) *algebraic-opt.* (strength reduction, ... ), (ii) *Operator-fusion* and (iii) *Operator-sinking*
	- **Dataflow-Level** optimisations include *dead-code*, *common-subexpression-elimination*, *static-memory-planning* and *memory-layout transformations*
- The **backend** of the compiler converts from *high-level IR* to *low-level IR*
	- Optimisations covered in paper - No notes from me...
- The **high-level IR** (Intermediate Representation) is used to establish the *control-flow* and *dependencies* of the program
	- The **DAG** (directed, acyclic Graph) based approach is one of the most traditional ways to model a program and is very intuitive
		- It has downsides, such as *semantic ambiguity*
	- The **let-binding** based approach uses a variable based approach and fixes some of the downsides of the DAG, however is less convenient to work with
	- Compilers usually represent the *tensor ops.* via (i) functions (XLA), (ii) Lambda Expressions, or (iii) Einstein notation
		- The set of supported ops usually includes *algebraic*, *neural-net-specific* (convolution), *tensor* (reshape, ...),  *reduction* (max) and *control flow*
		- Operations should also be differentiable
	- The compiler must know Layout, Size, ... of the *data* to properly optimise
		- *Placeholders* is data with known size, but no fixed pointer
		- Compilers can support unknown shapes, but can optimise for them less
- The **low-level IR** is used to generate the hardware specific code
	- **Halide**-based IR was originally developed for Image processing and aims to find a efficient sequence of computations
		- It is limited to rectangular access patterns - this isn't a problem most of the time in DL
	- **Polyhedral**-based IR optimises loop based flows with *polyhedral* access patterns (enabling non-rectangular accesses)
		- They are less integratable with many tuning mechanisms
	- There exist also many other methods, such as those employed by Glow and MLIR(LLVM-based)
	- While **JIT** compilers have more information about the runtime environment, **AOT** (Ahead of Time) compilers have a larger scope in static analysis
- The common DL compilers presented in the survey are **XLA**, **Glow**, **TVM**, **Tensor Compression (TC)**, and **nGraph**
- The authors highlighted the following topics as common shortcomings of current DL-Compilers
	- Good support for dynamic shapes
	- Advanced autotuning (many local minima may not be the the global minima)
	- Good support for Polyhedral modelling
	- Subgraph partitioning
	- Quantisation
	- Unified optimisations across different compilers
	- Privacy Protection support in the edge-cloud model