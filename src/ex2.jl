using StaticArrays, Distributions
using DistributedSparseGrids
import DistributedSparseGrids: AbstractCollocationPoint, AbstractHierarchicalCollocationPoint, AbstractHierarchicalSparseGrid, numlevels, coord, pt_idx, i_multi, level, scaling_weight, fval
using GLMakie, Colors, Printf
import Colors: distinguishable_colors, RGB, N0f8, colormap
import DistributedSparseGrids: generate_next_level, distributed_init_weights_inplace_ops
using LinearAlgebra
include("sparse_grid_visualisation.jl")

function lin_func(x,xmin,ymin,xmax,ymax)
    a = (ymax-ymin)/(xmax-xmin)
    b = ymax-a*xmax
    return a*x+b
end

function CPtoStoch(x::T1,dist::T2) where {T1<:Number, T2<:Truncated}
    return lin_func(x, -1.0, dist.lower, 1.0, dist.upper)
end

function sparse_grid(N::Int, pointprops; nlevel=6, RT=Matrix{Float64}, CT=Float64)
    # define collocation point
    CPType = CollocationPoint{N,CT}
    # define hierarchical collocation point
    HCPType = HierarchicalCollocationPoint{N,CPType,RT}
    # init grid
    asg = init(AHSG{N,HCPType},pointprops)
    #set of all collocation points
    cpts = Set{HierarchicalCollocationPoint{N,CPType,RT}}(collect(asg))
    # fully refine grid nlevel-1 times
    for i = 1:nlevel-1
        union!(cpts,generate_next_level!(asg))
    end
    return asg
end

# example borrowed from https://pde-on-gpu.vaw.ethz.ch/lecture1/#exercise_3_volcanic_bomb
const km_h   = 1000/3600
const nt_max = 100
const Δt     = 1.0
const g      = -9.81

cut_off = 2 # we cut off the normal distribution at +/-2 std. dev.
μ_v₀ = 120*km_h
σ_v₀ = 10*km_h
μ_α = 60.0
σ_α = 10.0
μ_𝐱₀ = 480
σ_𝐱₀ = 10
𝒩_v₀ = truncated(Normal(μ_v₀, σ_v₀), lower=μ_v₀-cut_off*σ_v₀, upper=μ_v₀+cut_off*σ_v₀)
𝒩_α = truncated(Normal(μ_α, σ_α), lower=μ_α-cut_off*σ_α, upper=μ_α+cut_off*σ_α)
𝒩_𝐱₀ = truncated(Normal(μ_𝐱₀, σ_𝐱₀), lower=μ_𝐱₀-cut_off*σ_𝐱₀, upper=μ_𝐱₀+cut_off*σ_𝐱₀)


function Position(𝛏, ID="", 𝒩_v₀=𝒩_v₀, 𝒩_α=𝒩_α, 𝒩_𝐱₀=𝒩_𝐱₀)
    v0 = CPtoStoch(𝛏[1], 𝒩_v₀)
    α = CPtoStoch(𝛏[2], 𝒩_α)
    𝐱0 = CPtoStoch(𝛏[3], 𝒩_𝐱₀)
    𝐯         = MVector{2, Float64}(undef)
    𝐯[1]      = v0*cosd(α)
    𝐯[2]      = v0*sind(α)
    𝐱         = zeros(2, nt_max)
    𝐱[:,1]   .= 𝐱0
    # Integrate trajectory
    for it=2:nt_max
        𝐯[2] = 𝐯[2] + g*Δt
        𝐱[1,it] = 𝐱[1,it-1] + 𝐯[1]*Δt
        𝐱[2,it] = 𝐱[2,it-1] + 𝐯[2]*Δt
        if 𝐱[2,it] < 0.0
            𝐱[1,it] = lin_func(0.0,𝐱[2,it-1],𝐱[1,it-1],𝐱[2,it],𝐱[1,it])
            𝐱[2,it] = 0.0
            break
        end 
    end
    
    return 𝐱    
end

function Cost(params, x_target)
    𝐱    = Position(params)
    idx  = findfirst(x->isapprox(x,0.0,atol=1e-3), 𝐱[2,:])
    cost = abs( 𝐱[1,idx-1] - x_target) 
    return cost, 𝐱, idx
end

x = rand(3)
τ = Position(x)
asg = sparse_grid(3, @SVector [1,1,1]; nlevel=6)
init_weights!(asg, Position)
comparefct(x) = norm(x)>1.0 # use only the first field as a refinement indicator
nrefsteps = 5
for i = 1:nrefsteps
    println("adaptive refinement step $i/$nrefsteps")
    cpts = generate_next_level!(asg, comparefct, 20)
    println("$(length(cpts)) new collocation points")
    init_weights!(asg, collect(cpts), Position)
end

GLMakie.activate!()
fig = Figure(size=(1200,800), fontsize=14)
view = fig[1,1] = GridLayout()
controls = fig[2,1] = GridLayout()
ax = Axis(view[1,1])
sl_x = Slider(controls[1, 1], range = -1.0:0.01:1.0, startvalue = 0.0, update_while_dragging=true, linewidth=14)
sl_y = Slider(controls[2, 1], range = -1.0:0.01:1.0, startvalue = 0.0, update_while_dragging=true, linewidth=14)
sl_z = Slider(controls[3, 1], range = -1.0:0.01:1.0, startvalue = 0.0, update_while_dragging=true, linewidth=14)
label1 = map!(Observable{Any}(),sl_x.value) do x
    _x = CPtoStoch(x,𝒩_v₀)
    @sprintf("v₀ = %.1f",_x)
end
label2 = map!(Observable{Any}(),sl_y.value) do x
    _x = CPtoStoch(x,𝒩_α)
    @sprintf("α = %.1f",_x)
end
label3 = map!(Observable{Any}(),sl_z.value) do x
    _x = CPtoStoch(x,𝒩_𝐱₀)
    @sprintf("𝐱₀ = %.1f",_x)
end
Label(controls[1, 2], label1)
Label(controls[2, 2], label2)
Label(controls[3, 2], label3)

_intp_res = interpolate(asg, [0.0,0.0,0.0])
intp_res = map!(Observable{Any}(),sl_x.value,sl_y.value,sl_z.value) do x,y,z
    global _intp_res
    interpolate!(_intp_res, asg, [x,y,z])
    return _intp_res
end
exact_res = map!(Observable{Any}(),sl_x.value,sl_y.value,sl_z.value) do x,y,z
    return Position((x,y,z))
end
scatter_x = map!(Observable{Any}(),intp_res) do 𝐱
    idx  = findfirst(x->isapprox(x,0.0,atol=1e-1), 𝐱[2,:])
    return 𝐱[1,1:idx]
end
scatter_y = map!(Observable{Any}(),exact_res) do 𝐱
    idx  = findfirst(x->isapprox(x,0.0,atol=1e-1), 𝐱[2,:])
    return 𝐱[2,1:idx]
end
exact_scatter_x = map!(Observable{Any}(),exact_res) do 𝐱
    idx  = findfirst(x->isapprox(x,0.0,atol=1e-1), 𝐱[2,:])
    return 𝐱[1,1:idx]
end
exact_scatter_y = map!(Observable{Any}(),intp_res) do 𝐱
    idx  = findfirst(x->isapprox(x,0.0,atol=1e-1), 𝐱[2,:])
    return 𝐱[2,1:idx]
end
scatterlines!(ax, scatter_x, scatter_y)
scatterlines!(ax, exact_scatter_x, exact_scatter_y)

display(fig)
   

    @info "Reference model landing x position: $(𝐱[1,idx-1]) "

    ### 2 - pertubation of 3 input parameter using a normal distribution 

    # Define perturbations of parameters
    ntest    = 20
    cost     = zeros(ntest)
    x_target = 210.0

    for test = 1:ntest
        params = ( 
            v0 = rand(Normal(120*km_h, 10*km_h)), 
            α  = rand(Normal(60.0,     10.0)),
            𝐱0 = @SVector[0.0, rand(Normal(480, 10))] 
        )
        cost[test], 𝐱, idx      = Cost(params, x_target)
        scatter!(ax, 𝐱[1,1:idx-1], 𝐱[2,1:idx-1])
    end
    ax  = Axis(f[2,1], xlabel="test", ylabel="cost")
    scatter!(ax, 1:ntest, cost)
    display(f)

#end