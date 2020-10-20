# Copyright (c) 2020: Matthew Wilhelm & Matthew Stuber.
# This work is licensed under the Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#############################################################################
# DynamicBoundspODEIneq.jl
# See https://github.com/PSORLab/DynamicBoundspODEIneq.jl
#############################################################################
# src/DynamicBoundspODEsIneq.jl
# main module file
#############################################################################

module DynamicBoundspODEsIneq

using DynamicBoundsBase, McCormick, ElasticArrays, DocStringExtensions
using ForwardDiff: Chunk, Dual, Partials, construct_seeds, single_seed
using DiffEqSensitivity: extract_local_sensitivities, ODEForwardSensitivityProblem
using DiffEqBase: remake, AbstractODEProblem, AbstractContinuousCallback, solve
using Sundials

const DBB = DynamicBoundsBase

import DynamicBoundsBase: relax!, set!, setall!, get, getall!, relax!
export DifferentialInequality, set!, setall!, get, getall!, relax!,
       DifferentialInequalityCond, DifferentialInequalityAffect,
       DifferentialInequalityAffectNeg, DifferentialInequalityf,
       LocalProblemStorage

import Base.size

include("local_integration_problem.jl")
include("contractors.jl")
include("DifferentialInequality.jl")

import DynamicBoundsBase: integrate!
export integrate!
include("integrate.jl")

end
