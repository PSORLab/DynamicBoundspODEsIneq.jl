"""
$(TYPEDEF)
"""
struct DifferentialInequalityCond <: Function
    nx::Int
    np::Int
    etol::Float64
end
function (d::DifferentialInequalityCond)(g, u::Vector{Float64}, t::Float64, integrator)::Nothing
    for i=1:d.nx
        g[i] = u[2*d.nx + i] - u[i] - integrator.p[d.np+i]*d.etol
        g[d.nx+i] = u[d.nx + i] - u[3*d.nx + i]  - integrator.p[d.np+d.nx+i]*d.etol
        integrator.p[d.np + 2*d.nx + i] = u[2*d.nx + i] - u[i]          # positive if cv > lo
        integrator.p[d.np + 3*d.nx + i] = u[d.nx + i] - u[3*d.nx + i]   # positive if hi > cc
    end
    return
end

"""
$(TYPEDEF)
"""
struct DifferentialInequalityAffect <: Function
    np::Int
    nx::Int
end
function (d::DifferentialInequalityAffect)(integrator, idx::Int)::Nothing
    integrator.p[d.np + idx] = 0.0
    return
end

"""
$(TYPEDEF)
"""
struct DifferentialInequalityAffectNeg <: Function
    np::Int
    nx::Int
end
function (d::DifferentialInequalityAffectNeg)(integrator, idx::Int)::Nothing
    @inbounds integrator.p[d.np+idx] = 1.0
    return
end

"""
$(TYPEDEF)
"""
struct DifferentialInequalityf{Z, F} <: Function
    f!::F
    nx::Int
    nm::Int
    np::Int
    p_mc::Vector{Z}
    P::Vector{Interval{Float64}}
    X::Vector{Interval{Float64}}
    x_mc::Vector{Z}
    xout_mc::Vector{Z}
    BetaL::Vector{Interval{Float64}}
    BetaU::Vector{Interval{Float64}}
    xout_intv1::Vector{Interval{Float64}}
    xout_intv2::Vector{Interval{Float64}}
    calculate_relax::Bool
    calculate_subgradient::Bool
    prng::UnitRange{Int64}
    xrng::UnitRange{Int64}
    polyhedral_constraint::PolyhedralConstraint
    has_apriori::Bool
    Xapriori::Vector{Interval{Float64}}
end
function DifferentialInequalityf(f!, Z, nx::Int, np::Int, P, relax::Bool, subgrad::Bool,
                    polyhedral_constraint, Xapriori)
     np = length(P)
     DifferentialInequalityf{Z,typeof(f!)}(f!, nx, size(polyhedral_constraint.A,1), np, zeros(Z,np), P,
                              zeros(Interval{Float64},nx), zeros(Z,nx), zeros(Z,nx),
                              zeros(Interval{Float64},nx), zeros(Interval{Float64},nx),
                              zeros(Interval{Float64},nx), zeros(Interval{Float64},nx),
                              relax, subgrad, 1:np, 1:nx, polyhedral_constraint, false, Xapriori)
end

function (d::DifferentialInequalityf{MC{N,T},F})(dx::Vector{Float64}, x::Vector{Float64},
                                     p::Vector{Float64}, t::Float64)::Nothing where {T<:RelaxTag, N, F}

    np = d.np
    nm = d.nm
    nx = d.nx

    if d.calculate_relax
        d.p_mc.= MC{N,T}.(p[1:np], d.P, d.prng)
    end

    for i=d.xrng
        d.X[i] = Interval{Float64}(x[i], x[nx + i])
        if d.calculate_relax
            is_cvmid::Bool = (x[i] <= x[2*nx + i] <= x[nx + i]) && d.calculate_subgradient
            is_ccmid::Bool = (x[i] <= x[3*nx + i] <= x[nx + i]) && d.calculate_subgradient
            cvmid::Float64, _ = mid3(x[i], x[nx + i], x[2*nx + i])
            ccmid::Float64, _ = mid3(x[i], x[nx + i], x[3*nx + i])
            if is_cvmid
                cvsgmid::SVector{N,Float64} = SVector{N}([x[4*nx + N*(i-1) + j] for j=d.prng])
            else
                cvsgmid = zero(SVector{N,Float64})
            end
            if is_ccmid
                ccsgmid::SVector{N,Float64} = SVector{N}([x[4*nx + N*(nx+i-1) + j] for j=d.prng])
            else
                ccsgmid = zero(SVector{N,Float64})
            end
            d.x_mc[i] = MC{N,T}(cvmid,  ccmid, d.X[i], cvsgmid, ccsgmid, (~is_cvmid || ~is_ccmid))
        end
    end
    d.BetaL .= d.X
    d.BetaU .= d.X

    if d.calculate_relax
        d.f!(d.xout_mc, d.x_mc, d.p_mc, t)
    end

    for i=d.xrng
        d.BetaL[i] = @interval(x[i])
        polyhedral_contact!(d.polyhedral_constraint, d.BetaL, d.Xapriori, nx, nm)
        d.f!(d.xout_intv1, d.BetaL, d.P, @interval(t))
        dx[i] = d.xout_intv1[i].lo

        d.BetaU[i] = @interval(x[nx+i])
        polyhedral_contact!(d.polyhedral_constraint, d.BetaU, d.Xapriori, nx, nm)
        d.f!(d.xout_intv2, d.BetaU, d.P, @interval(t))
        dx[nx+i] = d.xout_intv2[i].hi

        if d.calculate_relax
            if (dx[i] > d.xout_mc[i].cv) && (p[np + i] == 1.0)
                dx[2*nx+i] = dx[i]
                if d.calculate_subgradient
                    for j=d.prng
                        dx[4*nx+(i-1)*np+j] = 0.0
                    end
                end
            else
                dx[2*nx+i] = d.xout_mc[i].cv
                if d.calculate_subgradient
                    for j=d.prng
                        dx[4*nx+(i-1)*np+j] = d.xout_mc[i].cv_grad[j]
                    end
                end
            end
            if (dx[nx+i] < d.xout_mc[i].cc) && (p[np + nx + i] == 1.0)
                dx[3*nx+i] = dx[nx+i]
                if d.calculate_subgradient
                    for j=d.prng
                        dx[(4+np)*nx+(i-1)*np+j] = 0.0
                    end
                end
            else
                dx[3*nx+i] = d.xout_mc[i].cc
                if d.calculate_subgradient
                    for j=d.prng
                        dx[(4+np)*nx+(i-1)*np+j] = d.xout_mc[i].cc_grad[j]
                    end
                end
            end
        end
        d.BetaL[i] = d.X[i]
        d.BetaU[i] = d.X[i]
    end
    return
end

"""
$(TYPEDEF)

The DifferentialInequality type integrator represents the relaxed pODE problem as a
2*nx ODE problem is `calculate_relax` is `false`4*nx + 4*np*nx
- `x[1:nx]` are the lower interval bounds
- `x[(1+nx):2*nx]` are upper interval bounds
- `x[(1+2*nx):3*nx]` are convex relaxations (computed if calculate_relax = true)
- `x[(1+3*nx):4*nx]` are concave relaxations (computed if calculate_relax = true)
- `x[(1+4*nx):(4+np)*nx]` are subgradients of the convex relaxations (computed if calculate_subgradients = true)
- `x[(1+(4+np)*nx):(4+2*np)*nx]` are subgradients of the concave relaxations (computed if calculate_subgradients = true)
The first `np` parameter values correspond to the parameter values in the original
problem. If (calculate_relax = true) then `(np+1):(np+nx)` parameter values correspond
to the `b_i^c` variables used to detect a relaxation crossing a lower state bound.
The `(np+nx+1):(np+nx)` parameter values correspond to the `b_i^C` variables used
to detect a relaxation crossing a upper state bound. The variables are floats but
are valued 0.0 and 1.0 and can be intepreted as the corresponding 0-1 Boolean values.
Otherwise, only `np` parameter values are used.

$(TYPEDFIELDS)
"""
mutable struct DifferentialInequality{F, N, T<:RelaxTag, PRB1<:AbstractODEProblem, PRB2<:AbstractODEProblem,
                 INT1, CB<:AbstractContinuousCallback} <: AbstractODERelaxIntegator
    calculate_relax::Bool
    calculate_subgradient::Bool
    differentiable::Bool
    event_soft_tol::Float64
    p::Vector{Float64}
    pL::Vector{Float64}
    pU::Vector{Float64}
    p_mc::Vector{MC{N,T}}
    x0f::F
    x0::Vector{Float64}
    x0_mc::Vector{MC{N,T}}
    xL::ElasticArray{Float64,2,1}
    xU::ElasticArray{Float64,2,1}
    relax_ode_prob::PRB1
    relax_ode_integrator::INT1
    relax_t::Vector{Float64}
    relax_lo::ElasticArray{Float64,2,1}
    relax_hi::ElasticArray{Float64,2,1}
    relax_cv::ElasticArray{Float64,2,1}
    relax_cc::ElasticArray{Float64,2,1}
    relax_cv_grad::ElasticArray{SVector{N,Float64},2,1}
    relax_cc_grad::ElasticArray{SVector{N,Float64},2,1}
    relax_mc::ElasticArray{MC{N,T},2,1}
    vector_callback::CB
    integrator_state::IntegratorStates
    local_problem_storage::LocalProblemStorage{PRB2, N}
    np::Int
    nx::Int
    polyhedral_constraint::Union{PolyhedralConstraint,Nothing}
end
function DifferentialInequality(d::ODERelaxProb; calculate_relax::Bool = true,
                   calculate_subgradient::Bool = true,
                   differentiable::Bool = false,
                   event_soft_tol = 1E-4,
                   relax_ode_integrator = CVODE_Adams(),
                   local_ode_integrator = CVODE_Adams(),
                   user_t = Float64[])

    @assert ~calculate_subgradient || calculate_relax "Relaxations must be computed in order to compute subgradients"

    np = length(d.p)
    Z = differentiable ? MC{np,Diff} : MC{np,NS}
    xdim = calculate_relax ? 4*d.nx : 2*d.nx
    xdim += calculate_subgradient ? 2*d.nx*d.np : 0
    pdim = d.np + (calculate_relax ? 4*d.nx : 0)
    pL = d.pL
    pU = d.pU
    p = zeros(Float64, pdim)
    p[1:d.np] = 0.5*(d.pL + d.pU)

    utemp = zeros(Float64, xdim)
    x0temp = zeros(Float64, xdim)
    x0mctemp = zeros(Z, xdim)
    differentialInequalitycond! = DifferentialInequalityCond(d.nx, np, event_soft_tol)
    differentialInequalityaffect! = DifferentialInequalityAffect(np, d.nx)
    differentialInequalityaffectneg! = DifferentialInequalityAffectNeg(np, d.nx)
    vector_callback = VectorContinuousCallback(differentialInequalitycond!, differentialInequalityaffect!,
                                               2*d.nx; affect_neg! = differentialInequalityaffectneg!,
                                               rootfind=true, save_positions=(false,false),
                                               interp_points=20)

    P = Interval{Float64}.(pL, pU)
    p_mc = zeros(Z, np)
    const_bnds = d.constant_state_bounds # const_bnds = get(d, ConstantBounds())
    polyhedral_constraint = d.polyhedral_constraint #get(d, DenseLinearInvariant())
    if polyhedral_constraint !== nothing
        X_natural_box = Interval{Float64}.(const_bnds.xL, const_bnds.xU)
    else
        X_natural_box = fill(Interval{Float64}(-Inf,Inf), d.nx)
    end
    f = DifferentialInequalityf(d.f, Z, d.nx, np, P, calculate_relax, calculate_subgradient,
                   polyhedral_constraint, X_natural_box)

    relax_ode_prob = ODEProblem(f, utemp, d.tspan, p)
    keyword_integator = local_ode_integrator
    local_problem_storage = LocalProblemStorage(d, keyword_integator, user_t)

    relax_t = Float64[]
    relax_lo = ElasticArray(zeros(Float64,d.nx,2))
    relax_hi = ElasticArray(zeros(Float64,d.nx,2))
    relax_cv = ElasticArray(zeros(Float64,d.nx,2))
    relax_cc = ElasticArray(zeros(Float64,d.nx,2))
    relax_cv_grad = ElasticArray(zeros(SVector{np,Float64},d.nx,2))
    relax_cc_grad = ElasticArray(zeros(SVector{np,Float64},d.nx,2))
    relax_mc = ElasticArray(zeros(Z,d.nx,2))

    xL = ElasticArray(zeros(Float64,d.nx,2))
    xU = ElasticArray(zeros(Float64,d.nx,2))

    DifferentialInequality(calculate_relax, calculate_subgradient, differentiable,
              event_soft_tol, p, pL, pU, p_mc, d.x0, x0temp,
              x0mctemp, xL, xU, relax_ode_prob, relax_ode_integrator, relax_t,
              relax_lo, relax_hi, relax_cv, relax_cc, relax_cv_grad, relax_cc_grad,
              relax_mc, vector_callback, IntegratorStates(),
              local_problem_storage, np, d.nx, d.polyhedral_constraint)
end

function relax!(d::DifferentialInequality{F, N, T, PRB1, PRB2, INT1, CB}) where {F, N, T<:RelaxTag,
                                                                    PRB1<:AbstractODEProblem,
                                                                    PRB2<:AbstractODEProblem, INT1,
                                                                    CB<:AbstractContinuousCallback}
    # load functors
    if d.integrator_state.new_decision_pnt
        for i=1:d.np
            d.relax_ode_prob.f.f.P[i] = Interval{Float64}(d.pL[i], d.pU[i])
            d.p_mc[i] = MC{N, NS}(d.p[i], d.relax_ode_prob.f.f.P[i], i)
        end
        # populate initial condition
        d.x0_mc = d.x0f(d.p_mc)
        for i=1:d.nx
            d.x0[i] = d.x0_mc[i].Intv.lo
            d.x0[d.nx+i] = d.x0_mc[i].Intv.hi
            if d.calculate_relax
                d.x0[2*d.nx+i] = d.x0_mc[i].cv
                d.x0[3*d.nx+i] = d.x0_mc[i].cc
                if d.calculate_subgradient
                    for j=1:d.np
                        d.x0[4*d.nx + j + d.np*(i-1)] = d.x0_mc[i].cv_grad[j]
                        d.x0[(4+d.np)*d.nx + j + d.np*(i-1)] = d.x0_mc[i].cc_grad[j]
                    end
                end
            end
        end
        d.relax_ode_prob = remake(d.relax_ode_prob; u0=d.x0)
        if d.calculate_relax
            relax_ode_sol = solve(d.relax_ode_prob, d.relax_ode_integrator, callback = d.vector_callback, abstol = 1E-6, reltol = 1E-3)
        else
            relax_ode_sol = solve(d.relax_ode_prob, d.relax_ode_integrator, abstol = 1E-6, reltol = 1E-3)
        end
        relax_ode_sol_t = relax_ode_sol.t::Vector{Float64}
        new_length::Int = length(relax_ode_sol_t)
        resize!(d.relax_t, new_length)
        d.relax_t .= relax_ode_sol_t

        resize!(d.relax_lo, d.nx, new_length)
        resize!(d.relax_hi, d.nx, new_length)
        resize!(d.relax_cv, d.nx, new_length)
        resize!(d.relax_cc, d.nx, new_length)
        resize!(d.relax_cv_grad, d.nx, new_length)
        resize!(d.relax_cc_grad, d.nx, new_length)
        resize!(d.relax_mc, d.nx, new_length)

        for i in 1:new_length
            time_step_sol::Vector{Float64} = relax_ode_sol[i]
            for j in 1:d.nx
                step_lo::Float64 = time_step_sol[j]
                step_hi::Float64 = time_step_sol[d.nx+j]
                d.relax_lo[j,i] = step_lo
                d.relax_hi[j,i] = step_hi
                if d.calculate_relax
                    step_cv::Float64 = time_step_sol[2*d.nx+j]
                    step_cc::Float64 = time_step_sol[3*d.nx+j]
                    d.relax_cv[j,i] = step_cv
                    d.relax_cc[j,i] = step_cc
                    if d.calculate_subgradient
                        step_cv_grad::SVector{N,Float64} = SVector{N}(Float64[time_step_sol[4*d.nx + d.np*(j-1) + k] for k=1:d.np])
                        step_cc_grad::SVector{N,Float64} = SVector{N}(Float64[time_step_sol[4*d.nx + d.np*(d.nx+j-1) + k] for k=1:d.np])
                        d.relax_cv_grad[j,i] = step_cv_grad
                        d.relax_cc_grad[j,i] = step_cc_grad
                    else
                        step_cv_grad = zero(SVector{d.np,Float64})::SVector{N,Float64}
                        step_cc_grad = zero(SVector{d.np,Float64})::SVector{N,Float64}
                    end
                    d.relax_mc[j,i] = MC{N,NS}(step_cv, step_cc, Interval{Float64}(step_lo, step_hi),
                                               step_cv_grad, step_cc_grad, false)
                end
            end
        end
    end
    return
end

supports(::DifferentialInequality, ::IntegratorName) = true
supports(::DifferentialInequality, ::Gradient) = true
supports(::DifferentialInequality, ::Subgradient) = true
supports(::DifferentialInequality, ::Bound) = true
supports(::DifferentialInequality, ::Relaxation) = true
supports(::DifferentialInequality, ::IsNumeric) = true
supports(::DifferentialInequality, ::IsSolutionSet) = true
supports(::DifferentialInequality, ::TerminationStatus) = true
supports(::DifferentialInequality, ::Value) = true
supports(::DifferentialInequality, ::ParameterValue) = true

get(t::DifferentialInequality, v::IntegratorName) = "Scott 2013 Integrator"
get(t::DifferentialInequality, v::IsNumeric) = false
get(t::DifferentialInequality, v::IsSolutionSet) = true
get(t::DifferentialInequality, s::TerminationStatus) = t.termination_status

function getall!(out::Array{Float64,2}, t::DifferentialInequality, v::Value)
    out .= t.local_problem_storage.pode_x
    return
end

function getall!(out::Vector{Array{Float64,2}}, t::DifferentialInequality, g::Gradient{NOMINAL})
    for i in 1:t.np
        @inbounds for j in eachindex(out[i])
            out[i][j] = t.local_problem_storage.pode_dxdp[i][j]
        end
    end
    return
end

function getall!(out::Vector{Array{Float64,2}}, t::DifferentialInequality, g::Gradient{LOWER})
    if ~t.differentiable
        error("Integrator does not generate differential relaxations. Set the
               differentiable_flag field to true and reintegrate.")
    end
    for i in 1:t.np
        if t.evaluate_interval
            fill!(out[i], 0.0)
        else
            @inbounds for j in eachindex(out[i])
                out[i][j] = t.relax_cv_grad[j][i]
            end
        end
    end
    return
end
function getall!(out::Vector{Array{Float64,2}}, t::DifferentialInequality, g::Gradient{UPPER})
    if ~t.differentiable_flag
        error("Integrator does not generate differential relaxations. Set the
               differentiable_flag field to true and reintegrate.")
    end
    for i in 1:t.np
        if t.evaluate_interval
            fill!(out[i], 0.0)
        else
            @inbounds for j in eachindex(out[i])
                out[i][j] = t.relax_cc_grad[j][i]
            end
        end
    end
    return
end

function getall!(out::Vector{Array{Float64,2}}, t::DifferentialInequality, g::Subgradient{LOWER})
    for i in 1:t.np
        if t.evaluate_interval
            fill!(out[i], 0.0)
        else
            @inbounds for j in eachindex(out[i])
                out[i][j] = t.relax_cv_grad[j][i]
            end
        end
    end
    return
end
function getall!(out::Vector{Array{Float64,2}}, t::DifferentialInequality, g::Subgradient{UPPER})
    for i in 1:t.np
        if t.evaluate_interval
            fill!(out[i], 0.0)
        else
            @inbounds for j in eachindex(out[i])
                out[i][j] = t.relax_cc_grad[j][i]
            end
        end
    end
    return
end

function getall!(out::Array{Float64,2}, t::DifferentialInequality, v::Bound{LOWER})
    out .= t.relax_lo
    return
end

function getall!(out::Vector{Float64}, t::DifferentialInequality, v::Bound{LOWER})
    out[:] = t.relax_lo[1,:]
    return
end

function getall!(out::Array{Float64,2}, t::DifferentialInequality, v::Bound{UPPER})
    out .= t.relax_hi
    return
end

function getall!(out::Vector{Float64}, t::DifferentialInequality, v::Bound{UPPER})
    out[:] = t.relax_hi[1,:]
    return
end

function getall!(out::Array{Float64,2}, t::DifferentialInequality, v::Relaxation{LOWER})
    if t.evaluate_interval
        @inbounds for i in eachindex(out)
            out[i] = t.relax_lo[i]
        end
    else
        @inbounds for i in eachindex(out)
            out[i] = t.state_relax[i].cv
        end
    end
    return
end
function getall!(out::Vector{Float64}, t::DifferentialInequality, v::Relaxation{LOWER})
    if t.evaluate_interval
        @inbounds for i in eachindex(out)
            out[i] = t.X[i].lo
        end
    else
        @inbounds for i in eachindex(out)
            out[i] = t.state_relax[i].cv
        end
    end
    return
end

function getall!(out::Array{Float64,2}, t::DifferentialInequality, v::Relaxation{UPPER})
    if t.evaluate_interval
        @inbounds for i in eachindex(out)
            out[i] = t.X[i].hi
        end
    else
        @inbounds for i in eachindex(out)
            out[i] = t.state_relax[i].cc
        end
    end
    return
end
function getall!(out::Vector{Float64}, t::DifferentialInequality, v::Relaxation{UPPER})
    if t.evaluate_interval
        @inbounds for i in eachindex(out)
            out[i] = t.X[i].hi
        end
    else
        @inbounds for i in eachindex(out)
            out[i] = t.state_relax[i].cc
        end
    end
    return
end

function setall!(t::DifferentialInequality, v::ParameterBound{LOWER}, value::Vector{Float64})
    t.integrator_state.new_decision_box = true
    @inbounds for i in 1:t.np
        t.pL[i] = value[i]
    end
    return
end

function setall!(t::DifferentialInequality, v::ParameterBound{UPPER}, value::Vector{Float64})
    t.integrator_state.new_decision_box = true
    @inbounds for i in 1:t.np
        t.pU[i] = value[i]
    end
    return
end

function setall!(t::DifferentialInequality, v::ParameterValue, value::Vector{Float64})
    t.integrator_state.new_decision_pnt = true
    @inbounds for i in 1:t.np
        t.p[i] = value[i]
    end
    return
end

function setall!(t::DifferentialInequality, v::Bound{LOWER}, values::Array{Float64,2})
    if t.integrator_state.new_decision_box
        t.integrator_state.set_lower_state = true
    end
    for i in 1:t.nx
        @inbounds for j in 1:t.steps
            t.xL[i,j] = values[i,j]
        end
    end
    return
end

function setall!(t::DifferentialInequality, v::Bound{LOWER}, values::Vector{Float64})
    if t.integrator_state.new_decision_box
        t.integrator_state.set_lower_state = true
    end
    @inbounds for i in 1:t.steps
        t.xL[1,i] = values[i]
    end
    return
end

function setall!(t::DifferentialInequality, v::Bound{UPPER}, values::Array{Float64,2})
    if t.integrator_state.new_decision_box
        t.integrator_state.set_upper_state = true
    end
    for i in 1:t.nx
        @inbounds for j in 1:t.steps
            t.xU[i,j] = values[i,j]
        end
    end
    return
end

function setall!(t::DifferentialInequality, v::Bound{UPPER}, values::Vector{Float64})
    if t.integrator_state.new_decision_box
        t.integrator_state.set_upper_state = true
    end
    @inbounds for i in 1:t.steps
        t.xU[1,i] = values[i]
    end
    return
end