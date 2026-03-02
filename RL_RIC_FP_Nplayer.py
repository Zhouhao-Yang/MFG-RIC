"""
RL_RIC_FP_Nplayer.py
N-Player Randomized Impulse Control Game — Fictitious-Play Algorithm

Based on Algorithm 1 from "Mean Field Games for Randomized Impulse Control"
Extension to N symmetric players using Fictitious Play:
  - Single value network for player 1
  - All opponents imitate player 1's policy each iteration
  - Buffer collected once and reused across FP iterations

State dynamics (between interventions):
    dX^i_t = b_i(X^i_t) dt + sigma_i(X^i_t) dW^i_t,   i = 1,...,N

Symmetric parameters:
    b = 0,  sigma = sqrt(2)/2,  r = 0.5
    f_i(x) = h |x_i - mean(x)|,  h = 2
    l_i(xi) = K + k|xi|,  K = 3, k = 1
    psi_{i,j}(xi) = c = 1
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from copy import deepcopy
import math, os, sys, argparse

import torch, torch.nn as nn
import torch.optim as optim

# ============================================================
# Symmetric N-player parameters (PDF Section 2)
# ============================================================
r      = 0.5                    # discount rate  (all players)
b_drift = 0.0                  # drift          (all players)
sigma  = math.sqrt(2) / 2      # diffusion      (all players)
h_coef = 2.0                   # running-cost scaling
K_cost = 3.0                   # fixed intervention cost
k_cost = 1.0                   # proportional intervention cost
c_cost = 1.0                   # cross-player cost

x_min, x_max = -3.0, 3.0
xi_min, xi_max = -3.0, 3.0     # jump-size grid range

EPS = 1e-12
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================
# Cost helpers
# ============================================================
def f_run(x, N):
    """Running cost for player 1:
       f_1(x) = h |x_1 - mean(x)|.
       For N=2: h|x_1 - (x_1+x_2)/2| = (h/2)|x_1-x_2|.  (matches 2-player)"""
    mean_x = x.mean(dim=-1)                    # [B]
    return h_coef * torch.abs(x[:, 0] - mean_x)

def l_cost(xi):
    """Own impulse cost: l(xi) = K + k|xi|."""
    return K_cost + k_cost * torch.abs(xi)

def R_safe(pi):
    """Entropy regulariser R(pi) = pi - pi log(pi)."""
    return pi - pi.clamp_min(EPS) * torch.log(pi.clamp_min(EPS))

# ============================================================
# Value network  (N-D input -> scalar)
# ============================================================
class ValueNet(nn.Module):
    def __init__(self, n_players=2, width=128, depth=4):
        super().__init__()
        layers, d_in = [], n_players
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.Tanh()]
            d_in = width
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        self.n_players = n_players
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """x : [B, N] -> [B]"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)


def load_1d_diff_into_Nd(net_nd, path_1d, N):
    """Load a 1-D PsiNet checkpoint into an N-D ValueNet using
    the difference mapping V_Nd(x) ~ V_1d(x_1 - mean(x)).

    x_1 - mean(x) = x_1 (N-1)/N - (x_2+...+x_N)/N

    First layer:
        W_Nd[:, 0] =  W_1d[:, 0] * (N-1)/N
        W_Nd[:, j] = -W_1d[:, 0] / N            for j = 1..N-1
    All other layers: direct copy (shapes already match).

    For N=2 this reduces to:
        W[:,0] = W_1d/2,  W[:,1] = -W_1d/2   i.e. V((x1-x2)/2).
    """
    tgt_device = next(net_nd.parameters()).device
    sd_1d = torch.load(path_1d, map_location=tgt_device)
    if 'model_state' in sd_1d:
        sd_1d = sd_1d['model_state']
    sd_nd = net_nd.state_dict()

    for key in sd_nd:
        if key not in sd_1d:
            continue
        if sd_1d[key].shape == sd_nd[key].shape:
            sd_nd[key] = sd_1d[key]
        elif key == 'net.0.weight':          # first Linear layer
            w1d = sd_1d[key].squeeze(-1)     # [width]
            sd_nd[key][:, 0] = w1d * (N - 1) / N
            for j in range(1, N):
                sd_nd[key][:, j] = -w1d / N

    net_nd.load_state_dict(sd_nd)
    print(f"  Loaded 1-D diff checkpoint '{path_1d}' into {N}-D net "
          f"(x1 - mean(x) mapping)")
    return net_nd


# ============================================================
# M operator  (player 1's own randomised impulse)
# ============================================================
@torch.no_grad()
def M_lambda_op(phi_net, x_batch, lam2, N, m_samples=2048):
    """
    M_{lam2} phi(x) for player 1.

    Player 1 shifts x_1:
        -lam2 log E_{y~N(0,1)}[ exp(-(phi(y,x_2,...,x_N) + l(y-x_1)) / lam2) ]

    Columns 1..N-1 remain fixed; only column 0 is replaced.
    """
    B  = x_batch.size(0)
    x1 = x_batch[:, 0]                                        # [B]

    if lam2 >= 0.2:                         # --- soft-min (MC) ---
        y = torch.randn(m_samples, device=device)              # [M]
        y_e = y.unsqueeze(0).expand(B, m_samples)              # [B,M]

        # Build input: replace column 0 with y, keep columns 1..N-1
        inp = x_batch[:, None, :].expand(B, m_samples, N).clone()
        inp[:, :, 0] = y_e                                    # [B,M,N]

        jump = y_e - x1.unsqueeze(1)                           # [B,M]
        phi  = phi_net(inp.reshape(-1, N)).view(B, m_samples)
        ell  = l_cost(jump)
        G    = torch.exp(-(phi + ell) / lam2).mean(dim=1)
        return -lam2 * torch.log(G + EPS)

    else:                                   # --- classical min ---
        xi_g = torch.linspace(xi_min, xi_max, m_samples,
                              device=device)                   # [M]
        M = xi_g.size(0)

        new_x1 = x1.unsqueeze(1) + xi_g.unsqueeze(0)          # [B,M]

        inp = x_batch[:, None, :].expand(B, M, N).clone()
        inp[:, :, 0] = new_x1                                 # [B,M,N]

        phi = phi_net(inp.reshape(-1, N)).view(B, M)
        ell = l_cost(xi_g).unsqueeze(0)                        # [1,M]
        return (phi + ell).min(dim=1).values


# ============================================================
# H operator  (cross-player interaction, classical inf)
# ============================================================
@torch.no_grad()
def H_op(phi_net, x_batch, intervening_player, N, m_grid=500):
    """
    Cross-player operator with constant cost c.

    H^{j,1} V(x) = inf_y V(x_1,...,y_j,...,x_N) + c

    Sweeps column `intervening_player` over a grid, keeps others fixed.
    """
    B  = x_batch.size(0)
    yg = torch.linspace(x_min, x_max, m_grid, device=device)  # [G]
    G  = yg.size(0)

    inp = x_batch[:, None, :].expand(B, G, N).clone()         # [B,G,N]
    inp[:, :, intervening_player] = yg.unsqueeze(0).expand(B, G)

    phi = phi_net(inp.reshape(-1, N)).view(B, G)
    return phi.min(dim=1).values + c_cost


# ============================================================
# Trajectory simulation  (N-dim, pure diffusion, no jumps)
# ============================================================
def collect_buffer(n_steps, dt, batch_size, N):
    """Euler-Maruyama for N-D state. Returns [(state_t, state_{t+1}), ...].
    Collected ONCE before the FP loop and reused."""
    sqrt_dt = math.sqrt(dt)
    rho = torch.distributions.Uniform(x_min, x_max)
    X = rho.sample((batch_size, N)).to(device)
    buf = []
    for _ in range(n_steps):
        dW = torch.randn_like(X) * sqrt_dt
        X1 = X + b_drift * dt + sigma * dW
        buf.extend(zip(X.cpu(), X1.cpu()))
        X = X1
    return buf


# ============================================================
# One TD gradient step  (Fictitious Play — single player)
# ============================================================
def critic_step(v_net, v_frozen, N, lam1, lam2, dt,
                buffer, opt, minibatch=2048):
    """
    Semi-gradient TD update for player 1 under Fictitious Play.

    All N-1 opponents imitate player 1's policy (from frozen network).
    Single value network, single phase.

    TD equation:
        V(x) = e^{-r dt} s V(x') + dt [f_1
               + pi_1 M V̄  +  pi_opp sum_{j=2}^N H^{j,1} V̄
               - lam1 R(pi_1)]

    where s = (1 - pi_1 dt)(1 - pi_opp dt)^{N-1}.
    Both pi_1 and pi_opp are computed from the frozen network
    to prevent positive feedback loops within the inner loop.
    """
    if len(buffer) < 2:
        return 0.0

    mb  = min(minibatch, len(buffer))
    idx = np.random.choice(len(buffer), mb, replace=False)
    xt, xtp1 = zip(*[buffer[i] for i in idx])
    xt   = torch.stack(xt).to(device)       # [B, N]
    xtp1 = torch.stack(xtp1).to(device)

    # ---- value at current state (keeps gradient) ----
    v_t = v_net(xt)

    with torch.no_grad():
        v_tp1 = v_net(xtp1)

        # M for player 1 (frozen target)
        M_val = M_lambda_op(v_frozen, xt, lam2, N)

        # H operator for opponents.  V is symmetric in x_2,...,x_N
        # and the buffer coordinates are i.i.d., so every opponent's
        # H^j has the same expected value.  Compute once, scale by (N-1).
        H_single = H_op(v_frozen, xt, intervening_player=1, N=N)
        H_sum    = (N - 1) * H_single

        # Opponents' intensity from FROZEN network (FP: opponents' policy is fixed)
        v_frozen_val = v_frozen(xt)
        E_opp  = (M_val - v_frozen_val) / lam1
        pi_opp = torch.exp(torch.clamp(-E_opp, max=25.0)).clamp_max(1.0 / dt)

        # Player 1's intensity also from frozen network for stability
        # (makes inner loop pure policy-evaluation; policy updates at outer loop)
        pi_1 = pi_opp  # symmetric players, same frozen policy

        # survival factor: player 1 + (N-1) opponents, each with own intensity
        surv = ((1.0 - pi_1 * dt).clamp_min(0.0)
                * (1.0 - pi_opp * dt).clamp_min(0.0) ** (N - 1))
        # running cost for player 1
        f_val = f_run(xt, N)

    # TD error  (gradient only through v_t)
    # Player 1 intervenes with intensity pi_1, opponents with pi_opp
    td = (v_t
          - math.exp(-r * dt) * surv * v_tp1
          - dt * (f_val
                  + pi_1 * M_val
                  + pi_opp * H_sum
                  - lam1 * R_safe(pi_1)))
    loss = td.square().mean()

    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item(), pi_1, surv


# ============================================================
# Training driver  (Fictitious Play, N players)
# ============================================================
def train_fp_nplayer(N, lam1, lam2,
                     T=20.0, dt=0.02,
                     roll_batch=4096,
                     N_outer=30,
                     K_steps=400,
                     minibatch=4096, seed=0,
                     init_psi="random",
                     init_1p_path=None):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    n_steps = int(T / dt)

    # --- initialise value network (single network for player 1) ---
    v_net = ValueNet(n_players=N).to(device)

    if init_psi == "psi_1p":
        psi_path = init_1p_path or "pinn_models/psi_1p_rl.pt"
        load_1d_diff_into_Nd(v_net, psi_path, N)
    elif init_psi == "psi0":
        psi_1d_path = "pinn_models/psi_init_smooth.pt"
        load_1d_diff_into_Nd(v_net, psi_1d_path, N)
    elif init_psi == "psi_classical":
        psi_path = "pinn_models/psi_init_classical.pt"
        load_1d_diff_into_Nd(v_net, psi_path, N)
    else:  # "random"
        print("  Using random initialisation")

    opt = optim.AdamW(v_net.parameters(), lr=1e-3, weight_decay=1e-4)

    loss_hist = []

    # --- collect buffer ONCE (reused across all FP iterations) ---
    print(f"  Collecting trajectory buffer (N={N}, "
          f"batch={roll_batch}, steps={n_steps}) ...")
    buf = collect_buffer(n_steps, dt, roll_batch, N)
    print(f"  buffer size = {len(buf)}")

    for n in range(N_outer):
        print(f"\n{'='*60}")
        print(f"FP outer iteration {n}/{N_outer-1}  (N={N})")
        print(f"{'='*60}")

        # freeze target network
        v_net.eval()
        v_frozen = deepcopy(v_net).eval()

        # gradient steps on player 1's value network
        v_net.train()
        for k in range(K_steps):
            loss, pi_1, surv = critic_step(v_net, v_frozen, N,
                               lam1=lam1, lam2=lam2, dt=dt,
                               buffer=buf, opt=opt,
                               minibatch=minibatch)
            if k % 100 == 0 or k == K_steps - 1:
                print(f"  [FP  outer {n:2d}  inner {k:3d}]  TD-loss {loss:.3e}")
                print(f"    pi_1  min={pi_1.min().item():.4e}  max={pi_1.max().item():.4e}  mean={pi_1.mean().item():.4e}")
                print(f"    surv  min={surv.min().item():.4e}  max={surv.max().item():.4e}  mean={surv.mean().item():.4e}")
        loss_hist.append(loss)

        # ---- saves / plots every outer iteration ----
        _save_and_plot(v_net, v_frozen,
                       lam1, lam2, dt, n, seed, N,
                       loss_hist, init_psi, K_steps)

    # final checkpoint
    torch.save({"v": v_net.state_dict(),
                "opt": opt.state_dict(),
                "loss": loss_hist,
                "seed": seed, "init_psi": init_psi,
                "lam1": lam1, "lam2": lam2, "N": N,
                "K_steps": K_steps},
               f"TD_models_Np/final_N{N}_lam1_{lam1:.2f}_lam2_{lam2:.3f}"
               f"_K{K_steps}_initpsi_{init_psi}_seed{seed}.pt")

    with open(f"TD_loss_Np/losses_N{N}_lam1_{lam1:.2f}_lam2_{lam2:.3f}"
              f"_K{K_steps}_initpsi_{init_psi}_seed{seed}.pkl", "wb") as fp:
        pickle.dump({"loss": loss_hist, "K_steps": K_steps}, fp)

    return v_net.cpu()


# ============================================================
# Visualisation helpers
# ============================================================
def _save_and_plot(v, v_f, lam1, lam2, dt, it, seed, N,
                   lh, init_psi="random", K_steps=400):
    """Plot along the diagnostic line where mean(x)=0:
       x_1 = u/2,  x_j = -u/(2(N-1))  for j=2,...,N.
    For N=2 this reduces to the anti-diagonal x1=u/2, x2=-u/2."""
    Nu = 300
    u_vals = torch.linspace(2 * x_min, 2 * x_max, Nu, device=device)
    # Build diagnostic line:  x_1 = u/2,  x_j = -u / (2*(N-1))
    x1_line = u_vals / 2.0                                     # [Nu]
    xj_line = -u_vals / (2.0 * (N - 1))                       # [Nu]
    # Assemble [Nu, N] tensor
    line_pts = xj_line.unsqueeze(-1).expand(Nu, N).clone()     # [Nu, N]
    line_pts[:, 0] = x1_line

    with torch.no_grad():
        V_line   = v(line_pts).cpu().numpy()       # updated player 1
        Vf_line  = v_f(line_pts).cpu().numpy()     # frozen (opponents')

        # player 1 policy
        M_val  = M_lambda_op(v_f, line_pts, lam2, N)
        v_val  = v(line_pts)
        pi_line = torch.exp(torch.clamp(-(M_val - v_val) / lam1, max=25.0)
                            ).clamp_max(1.0 / dt).cpu().numpy()

    u_np = u_vals.cpu().numpy()
    tag = (f"FP_N{N}_lam1_{lam1:.2f}_lam2_{lam2:.3f}"
           f"_K{K_steps}_initpsi_{init_psi}_iter_{it}_seed_{seed}")

    # --- value functions (updated V^1 vs frozen V^{N\1}) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(u_np, V_line,  'b-', lw=2, label=r'$V^1$ (updated)')
    ax.plot(u_np, Vf_line, 'r--', lw=2, label=r'$V^{N\backslash 1}$ (frozen)')
    ax.set_xlabel(r'$u$ (diagnostic line, mean $= 0$)')
    ax.set_ylabel('Value')
    ax.set_title(f'Value functions  N={N}  iter {it}  '
                 rf'$\lambda_1$={lam1} $\lambda_2$={lam2}  '
                 f'init={init_psi}  seed={seed}')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"TD_figures_Np/values_{tag}.png", dpi=150)
    plt.close("all")

    # --- policy ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(u_np, pi_line, 'b-', lw=2, label=r'$\pi_1$')
    ax.set_xlabel(r'$u$ (diagnostic line, mean $= 0$)')
    ax.set_ylabel('Intensity')
    ax.set_title(f'Intervention intensity  N={N}  iter {it}  '
                 rf'$\lambda_1$={lam1} $\lambda_2$={lam2}  '
                 f'init={init_psi}  seed={seed}')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"TD_figures_Np/policies_{tag}.png", dpi=150)
    plt.close("all")

    # --- loss curve ---
    if len(lh) > 1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(lh, label='Player 1 (FP)')
        ax.set_xlabel('Outer iteration'); ax.set_ylabel('TD loss')
        ax.legend(); ax.grid(True)
        fig.suptitle(f'TD loss  N={N}  '
                     rf'$\lambda_1$={lam1} $\lambda_2$={lam2}  '
                     f'init={init_psi}  seed={seed}')
        plt.tight_layout()
        plt.savefig(f"TD_figures_Np/loss_{tag}.png", dpi=150)
        plt.close("all")

    # save model checkpoint
    torch.save({"v": v.state_dict(),
                "seed": seed, "init_psi": init_psi,
                "lam1": lam1, "lam2": lam2, "N": N, "iter": it,
                "K_steps": K_steps},
               f"TD_models_Np/ckpt_{tag}.pt")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    Path("TD_models_Np").mkdir(exist_ok=True)
    Path("TD_figures_Np").mkdir(exist_ok=True)
    Path("TD_loss_Np").mkdir(exist_ok=True)

    ap = argparse.ArgumentParser(
        description="N-Player Randomised Impulse Control — "
                    "Fictitious-Play algorithm")
    ap.add_argument("--N_players",  type=int,   default=2,
                    help="number of symmetric players")
    ap.add_argument("--init_psi",   type=str,   default="psi0",
                    choices=["random", "psi0", "psi_classical", "psi_1p"],
                    help="initialisation: random | psi0 | psi_classical | psi_1p")
    ap.add_argument("--init_1p_path", type=str, default=None,
                    help="path to 1-D checkpoint for psi_1p init "
                         "(default: pinn_models/psi_1p_rl.pt)")
    ap.add_argument("--lam1",       type=float, default=1.0)
    ap.add_argument("--lam2",       type=float, default=1.0)
    ap.add_argument("--N_outer",    type=int,   default=30)
    ap.add_argument("--K_steps",    type=int,   default=400)
    ap.add_argument("--roll_batch", type=int,   default=4096)
    ap.add_argument("--T",          type=float, default=20.0)
    ap.add_argument("--dt",         type=float, default=0.02)
    ap.add_argument("--minibatch",  type=int,   default=4096)
    ap.add_argument("--seed",       type=int,   default=0)
    args = ap.parse_args()

    print(f"\n=== {args.N_players}-Player FP Game: "
          f"lam1={args.lam1:.2f}  lam2={args.lam2:.3f} ===")
    print(f"Device: {device}")

    train_fp_nplayer(
        N=args.N_players,
        lam1=args.lam1,       lam2=args.lam2,
        T=args.T,             dt=args.dt,
        roll_batch=args.roll_batch,
        N_outer=args.N_outer,
        K_steps=args.K_steps,
        minibatch=args.minibatch, seed=args.seed,
        init_psi=args.init_psi,
        init_1p_path=args.init_1p_path)
