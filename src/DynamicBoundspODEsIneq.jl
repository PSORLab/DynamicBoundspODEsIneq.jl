module DynamicBoundspODEsIneq

using DynamicBoundsBase, DynamicBoundspODEs, McCormick, ElasticArrays, DocStringExtensions
using ForwardDiff: Chunk, Dual, Partials, construct_seeds, single_seed
using DiffEqSensitivity: extract_local_sensitivities, ODEForwardSensitivityProblem
using DiffEqBase: remake, AbstractODEProblem, AbstractContinuousCallback, solve
using Sundials

import DynamicBoundsBase: relax!, set, setall!, get, getall!, relax!
export Scott2013, set, setall!, get, getall!, relax!
include("local_integration_problem.jl")
include("contractors.jl")
include("Scott2013.jl")

import DynamicBoundsBase: integrate!
export integrate!
include("integrate.jl")

end # module
