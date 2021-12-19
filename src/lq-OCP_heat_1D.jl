using JuMP, Ipopt, LinearAlgebra

#A JuMP model is initialized
mod = Model(optimizer_with_attributes( Ipopt.Optimizer ,  "max_iter" => 100,
            "mumps_mem_percent" => 500))

#The parameters of the 1-D rod
L = 0.1                                 #Length of rod
λ = 45.0                                # Conductivity
c = 460.0                               # Specific heat capacitivity
ρ = 7800.0                              # Material density
α = λ / (c*ρ)                           # Diffusivity

#Number of spatial discretization points and length of each
N = 11                                 #Discretization points / nodes
Δx = L / (N-1)                         # Δx = x[i+1] - x[i]

#Integration from t0 = 0 to tf
tf = 900.0                             #Final time: 900 seconds = 15 minutes
Δt = 1.0                               #Sampling period
k = round(Int,tf / Δt);                #k = number of steps


#Initial and reference temperatures in Kelvin
θinit = 273.0
θref = 500.0

#bounded input values[u_min,u_max]
u_min = 0.0
u_max = 20e5; 

#Decision variables of the optimization problem
@variable(mod,θ[1:N,1:k]);                              # temperature at x[1:N] at time step k
@variable(mod,u_min <= u[1:k - 1] <= u_max);            # bounded constrained Input u
@variable(mod, y[1:k]);                                 # Output y

#Initial values
@constraint(mod, θ[:,1] .== θinit)
@constraint(mod, y[1] .== θinit)


#System dynamics
for j in 1:(k-1)

    for i in 2:N-1
        @constraint(mod,θ[i,j + 1] ==  θ[i,j] + Δt * α/( Δx^2)*(θ[i-1,j] - 2*θ[i,j] + θ[i+1,j]) ) # Diffusion at inner grid points
    end
    @constraint(mod, θ[1,j + 1] == (θ[1,j] + Δt*( α/(Δx^2) * (-2*θ[1,j] + 2*θ[2,j]) + 2/(c * ρ * Δx)*u[j]))) # Left side: heat input
    @constraint(mod, θ[end,j + 1] == ( θ[end,j] + Δt * α/(Δx^2) * (2*θ[end-1,j] - 2*θ[end,j]) )) # Right side: isolated side
    @constraint(mod,y[j + 1] ==  θ[end,j + 1]) # Measurement at left side
end

q_w = 1000; # Weighing coefficient for errors (or states)
r_w = 1e-4; # Weighing coefficient for input signals

# e(t) = r(t) - y(t) 
# err = q * sum( e(t_i)^2 )  with index i ∈ (1,k-1)
err = @NLexpression(mod, sum(q_w * (θref - y[j])^2 for j in 1:k-1) )

#in_err = r * sum( u(t_i)^2)
in_err = @NLexpression(mod, sum(r_w * u[j] ^ 2 for j in 1:k-1) )

# Cost function
J = @NLexpression(mod,0.5 * Δt * (err + in_err))

#defining the cost function
@NLobjective(mod, Min, J)

#optimizes the model
optimize!(mod)


# Plotting the results
using Plots
states_θstart = zeros(k)
states_θend = zeros(k)
input_u = zeros(k - 1)
for i = 1 : k-1
    states_θstart[i] = JuMP.value(θ[1,i])
    states_θend[i] = JuMP.value(θ[end,i])
    input_u[i] = JuMP.value(u[i])
end

states_θstart[end] = JuMP.value(θ[1,end])
states_θend[end] = JuMP.value(θ[end,end])


plot(states_θstart,label = "Left side")
plot!(states_θend,label = "Right side")

plot(input_u, label="Input")
