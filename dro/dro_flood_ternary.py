import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import ternary

# Parameters in Canadian dollars
num_transformers = 61
num_poles = 200
num_lines = 100  # Number of transmission lines between poles
line_length_meters = 50  # Average line length in meters
c_transformer = 25000  # Cost per transformer upgrade in CAD
c_pole = 2000          # Cost per pole reinforcement in CAD
c_line = 50 * line_length_meters  # Cost per line maintenance in CAD
eta = 350000           # Risk threshold in CAD
lambda_param = 0.7     # Weight on robustness
num_scenarios = 3      # High Wind, High Rain, High Flood

# Setup Ternary Plot Configurations
scale = 100  # Scaling for probability values
fig, tax = plt.subplots(subplot_kw=dict(projection='ternary'), figsize=(8, 6))
tax.set_title("Ternary Plot of Preventive Planning Costs under Environmental Scenarios")

# Initialize plot data for costs
cost_results = []

# Iterate over different probability combinations
for i in range(scale + 1):
    for j in range(scale + 1 - i):
        k = scale - i - j
        prob_scenarios = [i / scale, j / scale, k / scale]

        # Decision variables
        x_transformer = cp.Variable(num_transformers, boolean=True)
        x_pole = cp.Variable(num_poles, boolean=True)
        x_line = cp.Variable(num_lines, boolean=True)
        
        # CVaR and auxiliary variables
        gamma = cp.Variable(num_scenarios)
        
        # Objective function
        first_stage_cost = (
            c_transformer * cp.sum(x_transformer) + 
            c_pole * cp.sum(x_pole) + 
            c_line * cp.sum(x_line)
        )
        
        # Second-stage cost with CVaR weighting
        second_stage_cost = sum(
            prob_scenarios[s] * ((1 - lambda_param) * gamma[s] + lambda_param * (eta + (1 - lambda_param) / gamma[s])) 
            for s in range(num_scenarios)
        )
        total_cost = first_stage_cost + second_stage_cost
        
        # Constraints
        constraints = [
            x_transformer <= 1,  # Upgrade each transformer once (binary constraint)
            x_pole <= 1,         # Reinforce each pole once (binary constraint)
            x_line <= 1,         # Maintain each line once (binary constraint)
            gamma >= eta,        # Each CVaR value must meet the risk threshold `eta`
            gamma >= 0           # Non-negativity for gamma
        ]
        
        # Solve problem
        problem = cp.Problem(cp.Minimize(total_cost), constraints)
        problem.solve(solver=cp.GLPK_MI)
        
        # Store result for ternary plot
        if problem.status == cp.OPTIMAL:
            cost_results.append((prob_scenarios[0], prob_scenarios[1], prob_scenarios[2], problem.value))
            
# Convert results to ternary plot format
data = {(p[0], p[1], p[2]): p[3] for p in cost_results}

# Create ternary plot
fig, tax = plt.subplots(subplot_kw=dict(projection='ternary'), figsize=(8, 6))
tax.set_title("Preventive Planning Cost with Probabilistic Environmental Scenarios")
tax.heatmap(data, scale=scale, cmap="coolwarm", style="triangular", scientific=True)
tax.gridlines(color="gray", multiple=10)
tax.ticks(axis='lbr', multiple=10, linewidth=1)
tax.bottom_axis_label("High Wind")
tax.left_axis_label("High Rain")
tax.right_axis_label("High Flood")

plt.show()
