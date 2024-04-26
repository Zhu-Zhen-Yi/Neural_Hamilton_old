# Neural Hamilton

Use DeepONet to solve Hamilton equations.

## Tech stack

- `rugfield`: Rust library for generating Gaussian Random Fields

- `peroxide`: Rust library for solving ODE & save results as parquet file

- `polars`: Python library for reading parquet file

- `pytorch`: Python library for training DeepONet

- `wandb`: Python library for logging training process

## Problem description

- Input : Potential energy & Target points
  ```math
  V(x) = 2 - 16 \times \text{GRF}(x) \times x(x-1)^2, \quad t = [0,\,0.01,\,\cdots,\,1]
  ```
  ![potential.png](./figs/chebyshev_64_5_more/potential.png)

- Hamilton equation:
  ```math
  \begin{aligned}
  \dot{x}(t) &= p(t)\\
  \dot{p}(t) &= -\frac{\partial V}{\partial x}(x(t)) \\
  x(0) &= 0, \quad p(0) = 0
  \end{aligned}
  ```

- Output: Trajectory at target points
  ![trajectory.png](./figs/chebyshev_64_5_more/trajectory.png)

## Results

- For one of validation data
  ![potential_test.png](./figs/chebyshev_64_5_more/potential_test.png)

  ![trajectory_test.png](./figs/chebyshev_64_5_more/trajectory_test.png)

- Custom test data 1
  ```math
  \begin{aligned}
  &V(x) = 8 (x - 0.5)^2,~ x(0) = 0,~ x'(0) = 0 \\
  &x'' = -V'(x) = -16 (x - 0.5) \\
  \Rightarrow ~ &x(t) = 0.5 - 0.5 \cos(4t)
  \end{aligned}
  ```

  ![potential_pred.png](./figs/chebyshev_64_5_more/potential_pred.png)

  ![trajectory_pred.png](./figs/chebyshev_64_5_more/trajectory_pred.png)

- Custom test data 2
  ```math
  \begin{aligned}
  &V(x) = 4|x - 0.5|,~ x(0) = 0,~ x'(0) = 0 \\
  &x'' = -V'(x) = \begin{cases} 4, & x < 0.5 \\ -4, & x > 0.5 \end{cases} \\
  \Rightarrow ~ &x(t) = \begin{cases} 2t^2 & 0 < t < 0.5 \\ -2t^2 + 4t - 1& 0.5 \leq t < 1 \end{cases}
  \end{aligned}
  ```

  ![potential_pred2.png](./figs/chebyshev_64_5_more/potential_pred2.png)

  ![trajectory_pred2.png](./figs/chebyshev_64_5_more/trajectory_pred2.png)
