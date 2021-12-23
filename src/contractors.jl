# Copyright (c) 2020: Matthew Wilhelm & Matthew Stuber.
# This work is licensed under the Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#############################################################################
# DynamicBoundspODEIneq.jl
# See https://github.com/PSORLab/DynamicBoundspODEIneq.jl
#############################################################################
# src/contractors.jl
# Defines contractors (specifically the Ig contractor of Kai Shen and Joseph
# Scott, “Exploiting Nonlinear Invariants and Path Constraints to Achieve
# Tighter Reachable Set Enclosures Using Differential Inequalities”,
# Mathematics of Control, Signals, and Systems, In Press).
#############################################################################

const POLYHEDRON_A_TOL = 1e-4
const POLYHEDRON_WIDTH_TOL = 1e-12

function mid3_I(x::Interval{Float64}, y::Interval{Float64}, z::Interval{Float64})
    a = max(min(x,y), min(max(x,y),z))
    return a
end


"""
$(FUNCTIONNAME)

Contracts the state variable bounds based on the `PolyhedralConstraint`.
"""
function polyhedral_contact!(d::PolyhedralConstraint, Xin::Vector{Interval{Float64}},
                             Xp::Vector{Interval{Float64}}, nx::Int, nm::Int)
    X = Xin
    A = d.A
    b = d.b
    Aik = 0.0
    Aij = 0.0
    zL = Interval{Float64}(0.0)
    zU = Interval{Float64}(0.0)
    alphaL = Interval{Float64}(0.0)
    alphaU = Interval{Float64}(0.0)
    lambda = Interval{Float64}(0.0)
    gamma = Interval{Float64}(0.0)
    Xtemp = Interval{Float64}(0.0)
    for i = 1:nm
        alphaL = @interval(b[i])
        alphaU = @interval(b[i])
        for k = 1:nx
            Aik = A[i, k]
            zL = @interval(X[k].lo)
            zU = @interval(X[k].hi)
            alphaL -= max(Aik*zL, Aik*zU)
            alphaU -= min(Aik*zL, Aik*zU)
        end
        for j = 1:nx
            Aij = A[i,j]
            if abs(Aij) > POLYHEDRON_A_TOL
                zL = @interval(X[j].lo)
                zU = @interval(X[j].hi)
                alphaL += max(Aij*zL, Aij*zU)
                alphaU += min(Aij*zL, Aij*zU)
                lambda = min(alphaL/Aij, alphaU/Aij)
                gamma = max(alphaL/Aij, alphaU/Aij)
                zL = mid3_I(zL, zU, lambda)
                zU = mid3_I(zL, zU, gamma)
                Xtemp = @interval(zL, zU)
                if diam(Xtemp) > POLYHEDRON_WIDTH_TOL
                    X[j] = Xtemp
                end
                alphaL -= max(Aij*zL, Aij*zU)
                alphaU -= min(Aij*zL, Aij*zU)
            end
        end
    end
    Xin[:] = X[:]
    return
end

function polyhedral_contact!(d::Nothing, Xin::Vector{Interval{Float64}}, Xp::Vector{Interval{Float64}}, nx::Int, nm::Int)
    return
end
