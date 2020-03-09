using DynamicBoundspODEs, DynamicBoundspODEsIneq

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


tspan = (0.0,18.0e-5*250)
pL = [0.1; 0.033; 16.0; 5.0; 0.5; 0.3]
pU = 10.0*pL

prob = ODERelaxProb(f!, tspan, x0, pL, pU)

A = [0.0 -1.0 -1.0  0.0  0.0  0.0;
     0.0  0.0  0.0  0.0 -1.0 -1.0;
     1.0 -1.0 0.0 1.0 -1.0 0.0]
b = [-20.0; -16.0; -2.0]
set!(prob, PolyhedralConstraint(A, b))

xL = zeros(6)
xU = [34.0; 20.0; 20.0; 34.0; 16.0; 16.0]
set!(prob, ConstantStateBounds(xL,xU))

Scott2013(prob, calculate_relax = false, calculate_subgradient = false)
integrator = Scott2013(prob, calculate_relax = false, calculate_subgradient = false)

relax!(integrator)

ratio = rand(6)
pstar = pL.*ratio .+ pU.*(1.0 .- ratio)
setall!(integrator, ParameterValue(), pstar)
integrate!(integrator)
