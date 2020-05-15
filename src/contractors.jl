const POLYHEDRON_A_TOL = 1e-4
const POLYHEDRON_WIDTH_TOL = 1e-12

"""
$(FUNCTIONNAME)

Contracts the state variable bounds based on the `PolyhedralConstraint`.
"""
function polyhedral_contact!(d::PolyhedralConstraint, Xin::Vector{Interval{Float64}}, Xp::Vector{Interval{Float64}}, nx::Int, nm::Int)
    X = Xin
    A = d.A
    b = d.b
    Aik = 0.0
    Aij = 0.0
    bi = 0.0
    zL = 0.0
    zU = 0.0
    alphaL = 0.0
    alphaU = 0.0
    lambda = 0.0
    gamma = 0.0
    Xtemp = Interval{Float64}(0.0)
    for i = 1:nm
        alphaL = b[i]
        alphaU = b[i]
        for k = 1:nx
            Aik = A[i, k]
            zL = X[k].lo
            zU = X[k].hi
            alphaL -= max(Aik*zL, Aik*zU)
            alphaU -= min(Aik*zL, Aik*zU)
        end
        for j = 1:nx
            Aij = A[i,j]
            if abs(Aij) > POLYHEDRON_A_TOL
                zL = X[j].lo
                zU = X[j].hi
                alphaL += max(Aij*zL, Aij*zU)
                alphaU += min(Aij*zL, Aij*zU)
                lambda = min(alphaL/Aij, alphaU/Aij)
                gamma = max(alphaL/Aij, alphaU/Aij)
                zL, _ = mid3(zL, zU, lambda)
                zU, _ = mid3(zL, zU, gamma)
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
