# MFG-RIC: Reinforcement Learning for Mean Field Games of Randomized Impulse Control Games

This repository implements reinforcement-learning algorithms for **multi-player randomized impulse control games**, based on the two-fold randomization framework introduced in:

> **"Mean Field Games for Randomized Impulse Control"**

Players control jump processes in continuous time: each player chooses *when* to intervene (randomized via intensity `lambda_1`) and *where* to jump (randomized via softmin `lambda_2`), while paying running costs that depend on the relative positions of all players.

## Key Scripts

### `RL_RIC_TD_2player.py` — Two-Player TD Algorithm

Solves the symmetric 2-player game using **alternating best-response** updates:

- **Two separate value networks** `V^1`, `V^2` (one per player)
- **Phase I**: update player 1's value while freezing player 2's policy
- **Phase II**: update player 2's value using the newly updated player 1 policy
- Policies are derived from the randomized intervention operator `M_{lambda_2}`

```bash
python RL_RIC_TD_2player.py \
  --lam1 1.0 --lam2 1.0 \
  --init_psi psi0 \
  --N_outer 30 --K1_steps 400 --K2_steps 400 \
  --roll_batch 4096 --minibatch 4096 \
  --T 20.0 --dt 0.02 --seed 0
```

### `RL_RIC_FP_Nplayer.py` — N-Player Fictitious Play

Scales to **N symmetric players** using fictitious play:

- **Single value network** for player 1 (all opponents imitate player 1's policy)
- Exploits symmetry: the cross-player operator `H` is computed once and scaled by `N-1`
- Trajectory buffer collected once and reused across all outer FP iterations

```bash
python RL_RIC_FP_Nplayer.py \
  --N_players 3 \
  --lam1 1.0 --lam2 1.0 \
  --init_psi psi0 \
  --N_outer 30 --K_steps 400 \
  --roll_batch 4096 --minibatch 4096 \
  --T 20.0 --dt 0.02 --seed 0
```

## Supporting Scripts

| File | Description |
|------|-------------|
| `RL_RIC_TD.py` | Single-player randomized impulse control (1-D baseline) |
| `baseline_RIC.py` | Model-based policy iteration for the classical (non-randomized) problem |
| `train_pinn_2d_init.py` | Pre-trains PINN initializations for the value networks |
| `classical impulse control.ipynb` | Analytical solution to the classical 1-D impulse control problem |

## Model and Cost Parameters

All scripts use the same symmetric parameter set:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `b` | 0 | Drift |
| `sigma` | sqrt(2)/2 | Diffusion coefficient |
| `r` | 0.5 | Discount rate |
| `f(x)` | `h * \|x_i - mean(x)\|` | Running cost (h=2) |
| `l(xi)` | `K + k\|xi\|` | Own intervention cost (K=3, k=1) |
| `psi(xi)` | `c` | Cross-player intervention cost (c=1) |

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lam1` | 1.0 | Randomization parameter for intervention *timing* (higher = more random) |
| `--lam2` | 1.0 | Randomization parameter for jump *size* (higher = softer min) |
| `--init_psi` | `psi0` | Value network initialization: `random`, `psi0`, `psi_classical`, or `psi_1p` |
| `--N_outer` | 30 | Number of outer fixed-point / FP iterations |
| `--K_steps` | 400 | Inner gradient steps per outer iteration |
| `--dt` | 0.02 | Euler-Maruyama time step |

## Requirements

- Python 3.8+
- PyTorch
- NumPy, SciPy, Matplotlib
