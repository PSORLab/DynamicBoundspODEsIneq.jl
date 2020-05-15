#!/usr/bin/env julia
using Test, DynamicBoundspODEsIneq, DynamicBoundsBase

const DBB = DynamicBoundsBase

@testset "DifferentialInequality Relaxation" begin

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

    A = [0.0 -1.0 -1.0  0.0  0.0  0.0;
         0.0  0.0  0.0  0.0 -1.0 -1.0;
         1.0 -1.0 0.0 1.0 -1.0 0.0]
    b = [-20.0; -16.0; -2.0]
    set!(prob, PolyhedralConstraint(A, b))

    xL = zeros(6)
    xU = [34.0; 20.0; 20.0; 34.0; 16.0; 16.0]
    set!(prob, ConstantStateBounds(xL,xU))

    integrator = DifferentialInequality(prob, calculate_relax = true,
                                        calculate_subgradient = true)

    relax!(integrator)

    @test isapprox(integrator.relax_lo[6,77], 0.0169958, atol = 1E-5)
    @test isapprox(integrator.relax_hi[6,77], 8.08922, atol = 1E-3)
    @test isapprox(integrator.relax_cv[6,77], 0.0169958, atol = 1E-5)
    @test isapprox(integrator.relax_cc[6,77], 4.15395, atol = 1E-3)
    @test isapprox(integrator.relax_cv_grad[6,77][6], -0.00767, atol = 1E-5)
    @test isapprox(integrator.relax_cc_grad[6,77][5], 0.002858, atol = 1E-5)
end

@testset "Access functions" begin
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

    A = [0.0 -1.0 -1.0  0.0  0.0  0.0;
         0.0  0.0  0.0  0.0 -1.0 -1.0;
         1.0 -1.0 0.0 1.0 -1.0 0.0]
    b = [-20.0; -16.0; -2.0]
    set!(prob, PolyhedralConstraint(A, b))

    xL = zeros(6)
    xU = [34.0; 20.0; 20.0; 34.0; 16.0; 16.0]
    set!(prob, ConstantStateBounds(xL,xU))

    integrator = DifferentialInequality(prob, calculate_relax = true,
                                        calculate_subgradient = true)

    @test DBB.supports(integrator, DBB.IntegratorName())
    @test DBB.supports(integrator, DBB.Gradient())
    @test DBB.supports(integrator, DBB.Subgradient())
    @test DBB.supports(integrator, DBB.Bound())
    @test DBB.supports(integrator, DBB.Relaxation())
    @test DBB.supports(integrator, DBB.IsNumeric())
    @test DBB.supports(integrator, DBB.IsSolutionSet())
    @test DBB.supports(integrator, DBB.TerminationStatus())
    @test DBB.supports(integrator, DBB.Value())
    @test DBB.supports(integrator, DBB.ParameterValue())

    @test DBB.get(integrator, DBB.IntegratorName()) === "DifferentialInequality Integrator"
    @test !DBB.get(integrator, DBB.IsNumeric())
    @test DBB.get(integrator, DBB.IsSolutionSet())
    @test DBB.get(integrator, DBB.TerminationStatus()) === RELAXATION_NOT_CALLED

    #=
    out1 = []
    getall!(out1, integrator, Value())

    out2 = []
    getall!(out2, integrator, Gradient{Nominal}())

    out3 = []
    getall!(out3, integrator, Gradient{Lower}())

    out4 = []
    getall!(out4, integrator, Gradient{Upper}())

    out5 = []
    getall!(out5, integrator, Subgradient{Lower}())

    out6 = []
    getall!(out6, integrator, Subgradient{Upper}())

    out7 = []
    getall!(out7, integrator, Bound{Lower}())

    out8 = []
    getall!(out8, integrator, Bound{Upper}())

    out9 = []
    getall!(out9, integrator, Bound{Lower}())

    out10 = []
    getall!(out10, integrator, Bound{Upper}())

    out11 = []
    getall!(out11, integrator, Relaxation{Lower}())

    out12 = []
    getall!(out12, integrator, Relaxation{Upper}())

    out13 = []
    getall!(out13, integrator, ParameterBound{Lower}())

    out14 = []
    getall!(out14, integrator, ParameterBound{Upper}())

    val1 = []
    setall!(integrator, ParameterBound{Lower}(), val1)

    val2 = []
    setall!(integrator, ParameterBound{Upper}(), val2)

    out15 = []
    getall!(out15, integrator, ParameterValue())

    val3 = []
    setall!(integrator, ParameterValue(), val3)
    =#
end
