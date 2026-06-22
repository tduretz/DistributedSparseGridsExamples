using StaticArrays, CairoMakie, Distributions

# example borrowed from https://pde-on-gpu.vaw.ethz.ch/lecture1/#exercise_3_volcanic_bomb

const km_h   = 1000/3600
const nt_max = 100
const Δt     = 1.0
const g      = -9.81

function Position(params)
    v0, α, 𝐱0 = params
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
            break
        end 
    end
    return 𝐱    
end

function Cost(params, x_target)
    𝐱    = Position(params)
    idx  = findfirst(==(0), 𝐱[2,:])
    cost = abs( 𝐱[1,idx-1] - x_target) 
    return cost, 𝐱, idx
end

let 

    ### 1 - one reference model (useless, just for fun)

    # Call model for one set of parameters
    params = (v0=120*km_h, α=60.0, 𝐱0=@SVector[0.0, 480] ) 
    𝐱      = Position(params)

    f   = Figure()
    ax  = Axis(f[1,1], xlabel="x", ylabel="y")
    idx = findfirst(==(0), 𝐱[2,:])
    scatterlines!(ax, 𝐱[1,1:idx-1], 𝐱[2,1:idx-1])
    display(f)

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

end
