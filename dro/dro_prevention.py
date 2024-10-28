import cvxpy as cp
import numpy as np

# Define constants
num_scenarios = 4
num_transformers = 20
num_poles = 200
num_solar_panels = 100

# Scenario probabilities (to be changed by probability from data analytics)
probabilities = np.array([0.25, 0.25, 0.25, 0.25])

# Environmental conditions to be changed by probability from data analytics)
conditions = [
    {'wind': 1, 'rain': 3, 'flood': 4, 'storm': 4},
    {'wind': 2, 'rain': 1, 'flood': 3, 'storm': 2},
    {'wind': 3, 'rain': 2, 'flood': 1, 'storm': 4},
    {'wind': 1, 'rain': 1, 'flood': 1, 'storm': 1}
]

# Preventive planning costs
cost_upgrade_transformer = 50000  # Cost to upgrade one transformer
cost_reinforce_pole = 1000        # Cost to reinforce one pole
cost_maintenance_solar = 2000     # Cost for maintaining one rooftop solar

# Decision variables
x_transformers = cp.Variable(num_transformers, boolean=True)  # Upgrade decision for transformers
x_poles = cp.Variable(num_poles, boolean=True)  # Reinforce decision for poles
x_solar = cp.Variable(num_solar_panels, boolean=True)  # Maintenance decision for solar panels
gamma = cp.Variable(num_scenarios)  # CVaR variables for each scenario
eta = cp.Parameter(nonneg=True)  # Risk threshold (tunable parameter)

# Set a value for eta
eta.value = 50  #risk tolerance

# Minimize total cost + CVaR penalties
total_upgrade_cost = cost_upgrade_transformer * cp.sum(x_transformers)
total_reinforce_cost = cost_reinforce_pole * cp.sum(x_poles)
total_maintenance_cost = cost_maintenance_solar * cp.sum(x_solar)

# Define risk-aversion parameter
lambda_risk = 0.8  # to control risk aversion level

# Objective function incorporating CVaR
objective = cp.Minimize(
    total_upgrade_cost + total_reinforce_cost + total_maintenance_cost +
    (1 - lambda_risk) * cp.sum(cp.multiply(probabilities, gamma)) +
    lambda_risk * (eta + cp.sum(gamma))
)

constraints = []

# constraints
for i, cond in enumerate(conditions):
    # Preventive actions must ensure resilience based on environmental condition levels
    min_transformer_upgrades = int(num_transformers * (0.1 + 0.2 * (4 - cond['wind'])))
    min_pole_reinforcements = int(num_poles * (0.2 + 0.1 * (4 - cond['flood'])))
    min_solar_maintenance = int(num_solar_panels * (0.1 + 0.1 * (4 - cond['rain'])))

    constraints.append(cp.sum(x_transformers) >= min_transformer_upgrades)
    constraints.append(cp.sum(x_poles) >= min_pole_reinforcements)
    constraints.append(cp.sum(x_solar) >= min_solar_maintenance)

    # ($\gamma_s \geq \mathbb{E}_{P_s}[Q(x_1, u_s)] - \eta$)
    scenario_cost = (cost_upgrade_transformer * min_transformer_upgrades +
                     cost_reinforce_pole * min_pole_reinforcements +
                     cost_maintenance_solar * min_solar_maintenance)
    constraints.append(gamma[i] >= scenario_cost - eta)
    constraints.append(gamma[i] >= 0)

# Solve the optimization problem
problem = cp.Problem(objective, constraints)
result = problem.solve()

# Results
print("Optimal Preventive Planning Cost: $", result)
print("Transformers to Upgrade:", sum(x_transformers.value))
print("Poles to Reinforce:", sum(x_poles.value))
print("Solar Panels to Maintain:", sum(x_solar.value))
print("CVaR values per scenario:", gamma.value)
