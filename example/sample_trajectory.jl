# Copyright (c) 2020: Matthew Wilhelm & Matthew Stuber.
# This work is licensed under the Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#############################################################################
# DynamicBoundspODEIneq.jl
# See https://github.com/PSORLab/DynamicBoundspODEIneq.jl
#############################################################################
# example/sample_trajectory.jl
# A simple example illustrating how the underlying DifferentialInequality
# integrator may be used.
#############################################################################

# Imports Sundials and DifferentialEquations to access the CVODE solvers
# Imports DynamicBoundspODEsIneq to use DifferentialInequality integrator
# Imports DynamicBoundsBase for API and problem description functions.
using DynamicBoundspODEsIneq, DynamicBoundsBase, Sundials, DifferentialEquations

# Create problem
x0(p) = [34.0; 20.0; 0.0; 0.0; 16.0; 0.0]
function f!(du, u, p, t)
    du[1] = -p[1]*u[1]*u[2] + p[2]*u[3] + p[6]*u[6]
    du[2] = -p[1]*u[1]*u[2] + p[2]*u[3] + p[3]*u[3]
    du[3] =  p[1]*u[1]*u[2] - p[2]*u[3] - p[3]*u[3]
    du[4] =  p[3]*u[3] - p[4]*u[4]*u[5] + p[5]*u[6]
    du[5] = -p[4]*u[4]*u[5] + p[5]*u[6] + p[6]*u[6]
    du[6] =  p[4]*u[4]*u[5] - p[5]*u[6] - p[6]*u[6]
    return
end

tspan = (0.0,18.0e-5*50)
pL = [0.1; 0.033; 16.0; 5.0; 0.5; 0.3]
pU = 10.0*pL

prob = DynamicBoundsBase.ODERelaxProb(f!, tspan, x0, pL, pU)
set!(prob, SupportSet([i for i in 0.0:18.0e-5*50/10:18.0e-5*50]))

# Define a polyhedral constraint on state variables.
# In this case it arise from the stiochiometry of the
# underlying reacting system.
A = [0.0 -1.0 -1.0  0.0  0.0  0.0;
     0.0  0.0  0.0  0.0 -1.0 -1.0;
     1.0 -1.0 0.0 1.0 -1.0 0.0]
b = [-20.0; -16.0; -2.0]
set!(prob, PolyhedralConstraint(A, b))

# Provides apriori state bounds
xL = zeros(6)
xU = [34.0; 20.0; 20.0; 34.0; 16.0; 16.0]
set!(prob, ConstantStateBounds(xL, xU))

# Creates the integrator specifying that relaxations and subgradients
# thereof should be computed.
integrator = DifferentialInequality(prob, calculate_relax = true,
                                    calculate_subgradient = true,
                                    relax_ode_integrator = CVODE_Adams(),
                                    local_ode_integrator = CVODE_Adams())

# defines a point in p at which to compute relaxations
ratio = rand(6)
pstar = pL.*ratio .+ pU.*(1.0 .- ratio)
setall!(integrator, ParameterValue(), pstar)

# Computes the relaxation and subgradients
relax!(integrator)

# Computes a trajectory of the pODEs at pstar
integrate!(integrator)
