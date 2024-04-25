# Neural Hamilton

Using DeepONet to solve Hamilton equations.

## Problem description

- Input : Potential energy & Target points
  $$
  V(x) = 2 - 16 \times \text{GRF}(x) \times x(x-1)^2, \quad t = [0,\,0.01,\,\cdots,\,1]
  $$
  ![potential.png](./figs/potential.png)

- Hamilton equation:
  $$
  \begin{aligned}
  \dot{x}(t) &= p(t)\\
  \dot{p}(t) &= -\frac{\partial V}{\partial x}(x(t)) \\
  x(0) &= 0, \quad p(0) = 1
  \end{aligned}
  $$

- Output: Trajectory at target points
  ![trajectory.png](./figs/trajectory.png)

## Results

- For one of validation data
  ![potential_test.png](./figs/potential_test.png)

  ![trajectory_test.png](./figs/trajectory_test.png)

- Custom test data
  $$
  \begin{aligned}
  &V(x) = 8 (x - 0.5)^2,~ x(0) = 0,~ x'(0) = 0 \\
  &x'' = -V'(x) = -16 (x - 0.5) \\
  \Rightarrow ~ &x(t) = 0.5 - 0.5 \cos(4t)
  \end{aligned}
  $$

  ![potential_pred.png](./figs/potential_pred.png)

  ![trajectory_pred.png](./figs/trajectory_pred.png)
