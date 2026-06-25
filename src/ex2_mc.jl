using StaticArrays, Distributions
using GLMakie, Colors, Printf
using LinearAlgebra
using ProgressMeter

function lin_func(x,xmin,ymin,xmax,ymax)
    a = (ymax-ymin)/(xmax-xmin)
    b = ymax-a*xmax
    return a*x+b
end

function CPtoStoch(x::T1,dist::T2) where {T1<:Number, T2<:Truncated}
    return lin_func(x, -1.0, dist.lower, 1.0, dist.upper)
end

# example borrowed from https://pde-on-gpu.vaw.ethz.ch/lecture1/#exercise_3_volcanic_bomb
const km_h   = 1000/3600
const nt_max = 1000
const Δt     = 0.1
const g      = -9.81

cut_off = 4 # we cut off the normal distribution at +/-2 std. dev.
μ_v₀ = 120*km_h
σ_v₀ = 10*km_h
μ_α = 60.0
σ_α = 10.0
μ_𝐱₀ = 480
σ_𝐱₀ = 10
𝒩_v₀ = Distributions.truncated(Distributions.Normal(μ_v₀, σ_v₀), lower=μ_v₀-cut_off*σ_v₀, upper=μ_v₀+cut_off*σ_v₀)
𝒩_α = Distributions.truncated(Distributions.Normal(μ_α, σ_α), lower=μ_α-cut_off*σ_α, upper=μ_α+cut_off*σ_α)
𝒩_𝐱₀ = Distributions.truncated(Distributions.Normal(μ_𝐱₀, σ_𝐱₀), lower=μ_𝐱₀-cut_off*σ_𝐱₀, upper=μ_𝐱₀+cut_off*σ_𝐱₀)

MC_N = 1_500_000 # Number of Monte Carlo snapshots

function Position(𝛏)
    v0,α,𝐱0 = 𝛏
    𝐯         = MVector{2, Float64}(undef)
    𝐯[1]      = v0*cosd(α)
    𝐯[2]      = v0*sind(α)
    𝐱         = zeros(2, nt_max)
    𝐱[:,1]   .= 𝐱0
    idx = -1
    # Integrate trajectory
    for it=2:nt_max
        𝐯[2] = 𝐯[2] + g*Δt
        𝐱[1,it] = 𝐱[1,it-1] + 𝐯[1]*Δt
        𝐱[2,it] = 𝐱[2,it-1] + 𝐯[2]*Δt
        if isapprox(𝐱[2,it], 0.0, atol=1e-9)
            idx = it
            break
        elseif 𝐱[2,it] < 0.0
            idx = it
            𝐱[1,it] = lin_func(0.0,𝐱[2,it-1],𝐱[1,it-1],𝐱[2,it],𝐱[1,it])
            𝐱[2,it] = 0.0
            break
        end
        #if 𝐱[2,it] < 0.0
        #    idx = it-1
        #    break
        #end 
    end
    
    return 𝐱, idx    
end

function Cost(params, x_target)
    𝐱,idx    = Position(params)
    #idx  = findfirst(x->isapprox(x,0.0,atol=1e-3), 𝐱[2,:])
    cost = abs( 𝐱[1,idx-1] - x_target) 
    return cost, 𝐱, idx
end

function MC_sampling(sample_fun::F1, objective_fun::F2, N::Int) where {F1<:Function,F2<:Function}
    x = sample_fun()
    res = objective_fun(x)
    resvec = Vector{typeof(res)}(undef,N)
    println("Monte Carlo sampling")
    @showprogress Threads.@threads for i = 1:N
        resvec[i] = objective_fun(sample_fun())
    end
    return resvec
end
#sample_fun() = rand(3).*2.0.-1.0
# we have to sample with the measure of the normal distributions
sample_fun() = (rand(𝒩_v₀), rand(𝒩_α), rand(𝒩_𝐱₀))

MC_res = MC_sampling(sample_fun, Position, MC_N)

quantiles = [0.005,0.05, 0.15, 0.25, 0.75 ,0.85, 0.95, 0.995]
q_res_x = zeros(Float64,length(quantiles),nt_max)
q_res_y = zeros(Float64,length(quantiles),nt_max)
exp_val_res = zeros(Float64, 2, nt_max)
mc_div = zeros(Int64, 2, nt_max)

println("compute expected value")
@showprogress for i = 1:MC_N
    𝐱 = MC_res[i][1]
    it = MC_res[i][2]    
    exp_val_res[:,1:it] .+= 𝐱[:,1:it]
    mc_div[:,1:it] .+= 1
end
exp_val_res ./= mc_div

println("compute quantiles")
@showprogress Threads.@threads for val_i = 1:nt_max
    #quant_vals_x = Vector{Float64}(undef,MC_N)
    #quant_vals_y = Vector{Float64}(undef,MC_N)
    quant_vals_x = Vector{Float64}()
    quant_vals_y = Vector{Float64}()
    for i in 1:MC_N
        𝐱 = MC_res[i][1]
        it = MC_res[i][2]  
        if it >= val_i
            push!(quant_vals_x, 𝐱[1,val_i])
            push!(quant_vals_y, 𝐱[2,val_i])
        end
    end
    if length(quant_vals_x) > 0 && length(quant_vals_y)>0
        _qs_x = quantile(quant_vals_x, quantiles)
        _qs_y = quantile(quant_vals_y, quantiles)
        q_res_x[:,val_i] = _qs_x
        q_res_y[:,val_i] = _qs_y
    end
end 

using CairoMakie
using GeometryBasics
function plot_quantile!(ax, x1,y1,x2,y2,color,label)
    #idx1 = findfirst(x->isapprox(x,0.0,atol=1e-6), y1)
    #idx2 = findfirst(x->isapprox(x,0.0,atol=1e-6), y2)
    #pts = vcat(
    #    Point2f.(x1[1:idx1-1], y1[1:idx1-1]),
    #    reverse(Point2f.(x2[1:idx2-1], y2[1:idx2-1]))
    #)
    #poly!(ax, pts, color = color, strokewidth = 0)
    idx = findfirst(x->isapprox(x,0.0,atol=1e-6), y1)
    lines!(ax, x1[1:idx-1], y1[1:idx-1], color=color, linewidth=3)
    idx = findfirst(x->isapprox(x,0.0,atol=1e-6), y2)
    lines!(ax, x2[1:idx-1], y2[1:idx-1], color=color, label=label, linewidth=3)
    return nothing
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
    _x = CPtoStoch(x, 𝒩_v₀)
    @sprintf("v₀ = %.1f",_x)
end
label2 = map!(Observable{Any}(),sl_y.value) do x
    _x = CPtoStoch(x, 𝒩_α)
    @sprintf("α = %.1f",_x)
end
label3 = map!(Observable{Any}(),sl_z.value) do x
    _x = CPtoStoch(x, 𝒩_𝐱₀)
    @sprintf("𝐱₀ = %.1f",_x)
end
label4 = map!(Observable{Any}(),sl_x.value,sl_y.value,sl_z.value) do x,y,z
    _x = CPtoStoch(x, 𝒩_v₀)
    _y = CPtoStoch(y, 𝒩_α)
    _z = CPtoStoch(z, 𝒩_𝐱₀)
    P = pdf(𝒩_v₀, _x)*pdf(𝒩_α, _y)*pdf(𝒩_𝐱₀, _z)
    @sprintf("P(v₀=%.1f, α = %.1f, 𝐱₀ = %.1f) = %.3e",_x,_y,_z,P)
end
Label(controls[1, 2], label1)
Label(controls[2, 2], label2)
Label(controls[3, 2], label3)
Label(controls[4, 1:2], label4)
plot_quantile!(ax, q_res_x[1,:],q_res_y[1,:],q_res_x[end,:],q_res_y[end,:],RGBA(115/255, 147/255, 179/255,0.4), L"0.005 \leq P < 0.995")
plot_quantile!(ax, q_res_x[2,:],q_res_y[2,:],q_res_x[end-1,:],q_res_y[end-1,:],RGBA(250/255, 160/255, 160/255,0.4), L"0.05 \leq P < 0.95")
plot_quantile!(ax, q_res_x[3,:],q_res_y[3,:],q_res_x[end-2,:],q_res_y[end-2,:],RGBA(255/255, 234/255, 0/255,0.4), L"0.15 \leq P < 0.85")
plot_quantile!(ax, q_res_x[4,:],q_res_y[4,:],q_res_x[end-3,:],q_res_y[end-3,:],RGBA(80/255, 200/255, 120/255,0.4), L"0.25 \leq P < 0.75")
exact_res = map!(Observable{Any}(),sl_x.value,sl_y.value,sl_z.value) do x,y,z
    _x = CPtoStoch(x,𝒩_v₀)
    _y = CPtoStoch(y,𝒩_α)
    _z = CPtoStoch(z,𝒩_𝐱₀)
    return Position((_x,_y,_z))[1]
end
exact_scatter_x = map!(Observable{Any}(),exact_res) do 𝐱
    idx  = findfirst(x->isapprox(x,0.0,atol=1e-1), 𝐱[2,:])
    return 𝐱[1,1:idx]
end
exact_scatter_y = map!(Observable{Any}(),exact_res) do 𝐱
    idx  = findfirst(x->isapprox(x,0.0,atol=1e-1), 𝐱[2,:])
    return 𝐱[2,1:idx]
end
scatterlines!(ax, exact_scatter_x, exact_scatter_y, label=L"\text{Position}(\mathbf{v}_0,\,\alpha,\,\mathbf{x}_0)")
scatterlines!(ax, exp_val_res[1,:], exp_val_res[2,:], label=L"\mathbb{E}[x]")
axislegend()
display(fig)
