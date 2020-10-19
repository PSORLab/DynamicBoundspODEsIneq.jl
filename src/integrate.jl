# Copyright (c) 2020: Matthew Wilhelm & Matthew Stuber.
# This work is licensed under the Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#############################################################################
# DynamicBoundspODEIneq.jl
# See https://github.com/PSORLab/DynamicBoundspODEIneq.jl
#############################################################################
# src/integrate.jl
# Code used to integrate but not relax the underlying parametric ODE system.
#############################################################################

function integrate!(d::DifferentialInequality{F, N, T, PRB1, PRB2, INT1, CB}) where {F, N, T<:RelaxTag, PRB1<:AbstractODEProblem,
                                                          PRB2<:AbstractODEProblem, INT1,
                                                          CB<:AbstractContinuousCallback}

    d.local_problem_storage.pduals .= seed_duals(view(d.p, 1:N))
    d.local_problem_storage.x0duals .= d.x0f(d.local_problem_storage.pduals)
    for i = 1:N
        d.local_problem_storage.x0local[i] = d.local_problem_storage.x0duals[i].value
        for j = 1:d.nx
            d.local_problem_storage.x0local[(j + N + (i-1)*d.nx)] = d.local_problem_storage.x0duals[i].partials[j]
        end
    end
    d.local_problem_storage.pode_problem = remake(d.local_problem_storage.pode_problem,
                                                  u0 = d.local_problem_storage.x0local,
                                                  p = d.p[1:d.np])

    if ~isempty(d.local_problem_storage.user_t)
        solution = solve(d.local_problem_storage.pode_problem, d.local_problem_storage.integrator,
                     tstops = d.local_problem_storage.user_t, abstol = d.local_problem_storage.abstol,
                     adaptive = false, reltol=d.local_problem_storage.reltol)
    else
        solution = solve(d.local_problem_storage.pode_problem, d.local_problem_storage.integrator,
                         abstol = d.local_problem_storage.abstol, reltol=d.local_problem_storage.reltol)
    end

    x, dxdp = extract_local_sensitivities(solution)
    new_length = size(x,2)
    resize!(d.local_problem_storage.pode_x, d.nx, new_length)
    resize!(d.local_problem_storage.integrator_t, new_length)
    d.local_problem_storage.integrator_t .= solution.t
    d.local_problem_storage.pode_x .= x
    for i = 1:N
        resize!(d.local_problem_storage.pode_dxdp[i], d.nx, new_length)
        d.local_problem_storage.pode_dxdp[i] .= dxdp[i]
    end
    return
end
