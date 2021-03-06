# DynamicBoundspODEsIneq.jl
Differential Inequality Algorithms for DynamicBounds.jl

| **Linux/OS/Windows**                                                                     | **Coverage**                                             |                       
|:--------------------------------------------------------------------------------:|:-------------------------------------------------------:|
| [![Build Status](https://github.com/PSORLab/DynamicBoundspODEsIneq.jl/workflows/CI/badge.svg?branch=master)](https://github.com/PSORLab/DynamicBoundspODEsIneq.jl/actions?query=workflow%3ACI)  | [![codecov](https://codecov.io/gh/PSORLab/DynamicBoundspODEsIneq.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/PSORLab/DynamicBoundspODEsIneq.jl)                        

## Summary
This package implements a continuous-time differential inequality approach to
computing state bounds and relaxations using the DynamicBounds.jl framework. These methods solve auxiliary ODE/DAEs to generate relaxations of parametric pODES/pDAEs at specific points in time. Full documentation of this functionality may be found [here](https://psorlab.github.io/DynamicBounds.jl/dev/pODEsIneq/pODEsIneq) in the DynamicBounds.jl website.

## References
- JK Scott, PI Barton, *Bounds on the reachable sets of nonlinear control systems*,
  Automatica 49 (1), 93-100
- JK Scott, PI Barton, *Improved relaxations for the parametric solutions of ODEs using differential inequalities*, Journal of Global Optimization, 1-34
- JK Scott, *Reachability Analysis and Deterministic Global Optimization of Differential-Algebraic Systems*, Massachusetts Institute of Technology
- K Shen, JK Scott, *Rapid and accurate reachability analysis for nonlinear dynamic systems by exploiting model redundancy*, Computers & Chemical Engineering 106, 596-608
