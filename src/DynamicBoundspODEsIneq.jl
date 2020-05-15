module DynamicBoundspODEsIneq

using DynamicBoundsBase, McCormick, ElasticArrays, DocStringExtensions
using ForwardDiff: Chunk, Dual, Partials, construct_seeds, single_seed
using DiffEqSensitivity: extract_local_sensitivities, ODEForwardSensitivityProblem
using DiffEqBase: remake, AbstractODEProblem, AbstractContinuousCallback, solve
using Sundials

const DBB = DynamicBoundsBase

import DynamicBoundsBase: relax!, set!, setall!, get, getall!, relax!
export DifferentialInequality, set!, setall!, get, getall!, relax!
include("local_integration_problem.jl")
include("contractors.jl")
include("DifferentialInequality.jl")

import DynamicBoundsBase: integrate!
export integrate!
include("integrate.jl")

end # module
