# Imports Sundials and DifferentialEquations to access the CVODE solvers
# Imports DynamicBoundspODEsIneq to use DifferentialInequality integrator
# Imports DynamicBoundsBase for API and problem description functions.
using DynamicBoundspODEsIneq, DynamicBoundsBase, Sundials, DifferentialEquations

# Create problem
y0(u) = [1.2; 1.1]
function f!(dy, y, u, t)
    dy[1] = u[2]*y[1]*(one(typeof(u[1])) - y[2])*u[1]
    dy[2] = u[2]*y[2]*(y[1] - one(typeof(u[1])))
    nothing
end
tspan = (0.0, 1.0)

u_l = Float64[1.1; 2.95]
u_u = Float64[2.2; 3.05]

prob = DynamicBoundsBase.ODERelaxProb(f!, tspan, y0, u_l, u_u)
set!(prob, SupportSet([i for i in 0.0:0.01:1.0]))

# Creates the integrator specifying that relaxations and subgradients
# thereof should be computed.
integrator = DifferentialInequality(prob,
                                    calculate_relax = false,
                                    calculate_subgradient = false)

# defines a point in p at which to compute relaxations
ratio = rand(2)
pstar = u_l.*ratio .+ u_u.*(1.0 .- ratio)
setall!(integrator, ParameterValue(), pstar)

# Computes the relaxation and subgradients
relax!(integrator)

# Computes a trajectory of the pODEs at pstar
integrate!(integrator)
