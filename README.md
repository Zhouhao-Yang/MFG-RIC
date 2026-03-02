This repository contains the code for the paper:

> **“A Two-fold Randomization Framework for Impulse Control Problems”**

It includes:
- a classical analytic solver for a 1D cash–management impulse control problem,
- a model-based policy iteration baseline,
- a randomized RL/TD method (two-fold randomization) implemented with neural critics.


## RL_RIC_TD.py

This file assumes you already have:

- a **classical analytic value function** \(\psi_{\text{classical}}\) encoded via parameters saved to `classical_V.pkl`,
- pre–trained **PINN initializations** in `pinn_models/psi_init_smooth.pt` and `pinn_models/psi_init_classical.pt`.

The script:

1. Reconstructs the classical value function on a 1D grid;
2. Defines a neural network critic `PsiNet` to approximate the randomized value function;
3. Implements a **randomized nonlocal operator** \(\mathcal{N}^{\lambda_2}\) and the TD Bellman residual;
4. Runs an outer fixed–point loop with inner gradient steps to minimize TD error;
5. Logs the **relative \(L^2\) error** vs. the classical value and saves:
   - model checkpoints,
   - diagnostic plots of \(\psi_\theta\) vs. \(\psi_{\text{classical}}\),
   - a time series of relative \(L^2\) errors.


Run the script directly:

```bash
python train_randomised_td.py \
  --init_psi psi0 \
  --lam1 1.0 \
  --lam2 1.0 \
  --N_outer 30 \
  --gd_steps 400 \
  --roll_batch 4096 \
  --T 20.0 \
  --dt 0.02 \
  --minibatch 4096 \
  --seed 0
```

Important arguments:

- --init_psi {random,psi0,psi_classical}: initialization of iterated value function
- --lam1: randomization parameter for intervention time
- --lam2: randomization parameter for jump size
- --N_outer: number of outer fixed-point iterations
- --gd_steps: gradient steps per outer iteration


## classical impulse control.ipynb

This Jupyter notebook computes the analytical solution to the classical one-dimensional impulse control problem.


## baseline_RIC.py

This script implements the model-based policy iteration baseline for the classical impulse control problem.