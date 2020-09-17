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

    relax!(integrator)
    integrate!(integrator)

    t = integrator
    len0 = length(t.relax_lo)
    len1 = length(t.relax_hi)
    len3 = length(t.relax_cv_grad)
    len4 = length(t.relax_cc_grad)
    len5 = length(t.local_problem_storage.pode_x)
    len5a = length(t.local_problem_storage.pode_x[1])
    len6 = length(t.local_problem_storage.pode_dxdp[1])
    println("len0 = $len0")
    println("len1 = $len1")
    println("len3 = $len3")
    println("len4 = $len4")
    println("len5 = $len5, len5a = $(len5a)")
    println("len6 = $len6")

    #relax_lo = integrator.relax_lo
    #relax_hi = integrator.relax_hi
    #relax_cv_grad = integrator.relax_cv_grad
    #relax_cc_grad = integrator.relax_cc_grad
    #pode_x = integrator.local_problem_storage.pode_x
    #pode_dxdp1 = integrator.local_problem_storage.pode_dxdp[1]

    vout = zeros(6, size(t.local_problem_storage.pode_x, 2))
    DBB.getall!(vout, integrator, Value())
    @show vout[5,end] == 15.388489679114812
    @show vout[6,end] == 0.6115103208851901

    out2 = Matrix{Float64}[]
    for i = 1:6
        push!(out2, zeros(6, size(t.local_problem_storage.pode_x, 2)))
    end
    getall!(out2, integrator, Gradient{Nominal}())

    #=
    out3 = Matrix{Float64}[]
    for i = 1:6
        push!(out3, zeros(6, size(t.local_problem_storage.pode_x, 2)))
    end
    getall!(out3, integrator, Gradient{Lower}())

    out4 = Matrix{Float64}[]
    for i = 1:6
        push!(out4, zeros(6, size(t.local_problem_storage.pode_x, 2)))
    end
    getall!(out4, integrator, Gradient{Upper}())

    out5 = []
    for i = 1:6
        push!(out5, zeros(6, size(t.local_problem_storage.pode_x, 2)))
    end
    getall!(out5, integrator, Subgradient{Lower}())

    out6 = []
    for i = 1:6
        push!(out6, zeros(6, size(t.local_problem_storage.pode_x, 2)))
    end
    getall!(out6, integrator, Subgradient{Upper}())
    =#

    out7 = zeros(6, size(t.relax_lo,2))
    DBB.getall!(out7, integrator, DBB.Bound{Lower}())
    @test out7[6, 77] == 0.01699582373787887

    out8 = zeros(6, size(t.relax_lo,2))
    DBB.getall!(out8, integrator, DBB.Bound{Upper}())
    @test out8[6, 77]== 8.089220917692634

    out11 = zeros(6, size(t.relax_lo,2))
    DBB.getall!(out11, integrator, DBB.Relaxation{Lower}())
    @test out11[6, 77] == 0.01699582373787887

    out12 = zeros(6, size(t.relax_lo,2))
    DBB.getall!(out12, integrator, DBB.Relaxation{Upper}())
    @test out12[6, 77] == 4.153947135290472

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
end
