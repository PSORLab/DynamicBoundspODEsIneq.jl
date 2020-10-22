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
    if !d.calculate_local_sensitivity
        if length(d.local_problem_storage.x0local) != d.nx
            resize!(d.local_problem_storage.x0local, d.nx)
        end
    else
        if length(d.local_problem_storage.x0local) != d.nx*(d.np + 1)
            resize!(d.local_problem_storage.x0local, d.nx*(d.np + 1))
        end
    end
    for i = 1:d.nx
        d.local_problem_storage.x0local[i] = d.local_problem_storage.x0duals[i].value
        if d.calculate_local_sensitivity
            for j = 1:N
                d.local_problem_storage.x0local[(j + d.nx + (i-1)*d.nx)] = d.local_problem_storage.x0duals[i].partials[j]
            end
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

    new_length = length(solution.t)
    if d.calculate_local_sensitivity
        x, dxdp = extract_local_sensitivities(solution)
    else
        x = solution.u
    end

    resize!(d.local_problem_storage.pode_x, d.nx, new_length)
    resize!(d.local_problem_storage.integrator_t, new_length)
    prior_length = length(d.local_problem_storage.integrator_t)
    if new_length == prior_length
        d.local_problem_storage.integrator_t .= solution.t
    else
        d.local_problem_storage.integrator_t = solution.t
    end

    for i = 1:new_length
        d.local_problem_storage.pode_x[:,i] .= x[i]
    end
    if d.calculate_local_sensitivity
        for i = 1:N
            resize!(d.local_problem_storage.pode_dxdp[i], d.nx, new_length)
            d.local_problem_storage.pode_dxdp[i] .= dxdp[i]
        end
    end

    empty!(d.local_t_dict_flt)
    empty!(d.local_t_dict_indx)

    for (tindx, t) in enumerate(solution.t)
        d.local_t_dict_flt[t] = tindx
    end

    if !isempty(d.local_problem_storage.user_t)
        next_support_time = d.local_problem_storage.user_t[1]
        supports_left = length(d.local_problem_storage.user_t)
        loc_count = 1
        for (tindx, t) in enumerate(solution.t)
            if t == next_support_time
                d.local_t_dict_indx[loc_count] = tindx
                loc_count += 1
                supports_left -= 1
                if supports_left > 0
                    next_support_time = d.local_problem_storage.user_t[loc_count]
                end
            end
        end
    end

    return
end
