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

    DBB.relax!(integrator)
    DBB.integrate!(integrator)

    t = integrator

    #support_set = DBB.get(integrator, DBB.SupportSet())
    #@test support_set[3]

    vout = zeros(6, size(t.local_problem_storage.pode_x, 2))
    DBB.getall!(vout, integrator, Value())
    @test isapprox(vout[5,end], 15.388489679114812, atol = 1E-8)
    @test isapprox(vout[6,end], 0.6115103208851901, atol = 1E-8)

    out2 = Matrix{Float64}[]
    for i = 1:6
        push!(out2, zeros(6, size(t.local_problem_storage.pode_x, 2)))
    end
    DBB.getall!(out2, integrator, DBB.Gradient{Nominal}())
    @test isapprox(out2[2][3, 20], -0.00015998255331242103, atol = 1E-8)

    out3 = Matrix{Float64}[]
    for i = 1:6
        push!(out3, zeros(6, size(t.local_problem_storage.pode_x, 2)))
    end
    @test_throws ErrorException DBB.getall!(out3, integrator, DBB.Gradient{Lower}())

    out4 = Matrix{Float64}[]
    for i = 1:6
        push!(out4, zeros(6, size(t.local_problem_storage.pode_x, 2)))
    end
    @test_throws ErrorException DBB.getall!(out4, integrator, DBB.Gradient{Upper}())

    out5 = Matrix{Float64}[]
    for i = 1:6
        push!(out5, zeros(6, size(t.local_problem_storage.pode_x, 2)))
    end
    DBB.getall!(out5, integrator, Subgradient{Lower}())
    @test isapprox(out5[2][3, 20], -0.0009550904931136906, atol = 1E-8)

    out6 = Matrix{Float64}[]
    for i = 1:6
        push!(out6, zeros(6, size(t.local_problem_storage.pode_x, 2)))
    end
    DBB.getall!(out6, integrator, DBB.Subgradient{Upper}())
    @test isapprox(out6[2][3, 20], -7.440794834434013e-5, atol = 1E-8)

    out7 = zeros(6, size(t.relax_lo,2))
    DBB.getall!(out7, integrator, DBB.Bound{Lower}())
    @test isapprox(out7[6, 77], 0.01699582373787887, atol = 1E-8)

    out8 = zeros(6, size(t.relax_lo,2))
    DBB.getall!(out8, integrator, DBB.Bound{Upper}())
    @test isapprox(out8[6, 77], 8.089220917692634, atol = 1E-8)

    out11 = zeros(6, size(t.relax_lo,2))
    DBB.getall!(out11, integrator, DBB.Relaxation{Lower}())
    @test isapprox(out11[6, 77], 0.01699582373787887, atol = 1E-8)

    out12 = zeros(6, size(t.relax_lo,2))
    DBB.getall!(out12, integrator, DBB.Relaxation{Upper}())
    @test isapprox(out12[6, 77], 4.153947135290472, atol = 1E-8)

    val1 = zeros(6) .- 0.1
    DBB.setall!(integrator, DBB.ParameterBound{Lower}(), val1)
    @test integrator.pL[1] == -0.1

    val2 = zeros(6) .+ 0.2
    DBB.setall!(integrator, DBB.ParameterBound{Upper}(), val2)
    @test integrator.pU[1] == 0.2

    val3 = zeros(6) .+ 0.1
    DBB.setall!(integrator, DBB.ParameterValue(), val3)
    @test integrator.p[1] == 0.1

    val01 = -0.3
    DBB.set!(integrator, DBB.ParameterBound{Lower}(2), val01)
    @test integrator.pL[2] == -0.3

    val02 = 0.5
    DBB.set!(integrator, DBB.ParameterBound{Upper}(2), val02)
    @test integrator.pU[2] == 0.5

    val03 = 0.2
    DBB.set!(integrator, DBB.ParameterValue(2), val03)
    @test integrator.p[2] == 0.2

    #out13 = zeros(size(t.relax_cv,2))
    #DBB.getall!(out13, integrator, DBB.Relaxation{Lower}())
    #@test isapprox(out13[12], 33.75044934920534, atol = 1E-8)

    #out14 = zeros(size(t.relax_cc,2))
    #DBB.getall!(out14, integrator, DBB.Relaxation{Upper}())
    #@test isapprox(out14[12], 33.75384294104528, atol = 1E-8)

    out15 = zeros(size(t.relax_lo,2))
    DBB.getall!(out15, integrator, DBB.Bound{Lower}())
    @test isapprox(out15[12], 33.552829768732664, atol = 1E-8)

    out16 = zeros(size(t.relax_lo,2))
    DBB.getall!(out16, integrator, DBB.Bound{Upper}())
    @test isapprox(out16[12], 33.95462900665603, atol = 1E-8)
end
