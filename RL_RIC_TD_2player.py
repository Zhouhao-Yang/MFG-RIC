"""
RL_RIC_TD_2player.py
Two-Player Randomized Impulse Control Game — TD-Error Algorithm

Based on Algorithm 1 from "Mean Field Games for Randomized Impulse Control"
Section 2: Symmetric 2-player case.

State dynamics (between interventions):
    dX^i_t = b_i(X^i_t) dt + sigma_i(X^i_t) dW^i_t,   i = 1,2

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
# Symmetric 2-player parameters (PDF Section 2)
# ============================================================
r      = 0.5                    # discount rate  (r_1 = r_2)
b_drift = 0.0                  # drift          (b_1 = b_2)
sigma  = math.sqrt(2) / 2      # diffusion      (sigma_1 = sigma_2)
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
def f_run(x1, x2):
    """Running cost for player i (symmetric):
       f_i = h |x_i - (x_1+x_2)/2| = (h/2)|x_1-x_2|.
       With h=2 this gives |x_1-x_2|."""
    return (h_coef / 2.0) * torch.abs(x1 - x2)

def l_cost(xi):
    """Own impulse cost: l(xi) = K + k|xi|."""
    return K_cost + k_cost * torch.abs(xi)

def R_safe(pi):
    """Entropy regulariser R(pi) = pi - pi log(pi)."""
    return pi - pi.clamp_min(EPS) * torch.log(pi.clamp_min(EPS))

# ============================================================
# Value network  (2-D input -> scalar)
# ============================================================
class ValueNet(nn.Module):
    def __init__(self, width=128, depth=4):
        super().__init__()
        layers, d_in = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.Tanh()]
            d_in = width
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """x : [B, 2] -> [B]"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)


def load_1d_into_2d(net_2d, path_1d, player):
    """Load a 1-D PsiNet checkpoint into a 2-D ValueNet.

    Only the first Linear layer differs in shape:
        PsiNet  layer 0 weight: [width, 1]
        ValueNet layer 0 weight: [width, 2]

    Strategy: copy the 1-D weight into the column for the
    player's own coordinate (col 0 for player 1, col 1 for
    player 2) and zero the other column.  All remaining layers
    are shape-compatible and copied directly.
    """
    sd_1d = torch.load(path_1d, map_location='cpu')
    # handle checkpoints that wrap state_dict in a dict
    if 'model_state' in sd_1d:
        sd_1d = sd_1d['model_state']
    sd_2d = net_2d.state_dict()

    for key in sd_2d:
        if key not in sd_1d:
            continue
        if sd_1d[key].shape == sd_2d[key].shape:
            sd_2d[key] = sd_1d[key]
        elif key == 'net.0.weight':          # first Linear layer
            col = 0 if player == 1 else 1
            sd_2d[key] = torch.zeros_like(sd_2d[key])
            sd_2d[key][:, col] = sd_1d[key].squeeze(-1)
        # bias of first layer has the same shape → already handled

    net_2d.load_state_dict(sd_2d)
    print(f"  Loaded 1-D checkpoint '{path_1d}' into player {player} "
          f"(own coord = col {0 if player==1 else 1})")
    return net_2d


def load_1d_diff_into_2d(net_2d, path_1d):
    """Load a 1-D PsiNet checkpoint into a 2-D ValueNet using
    the difference mapping V_2d(x1,x2) = V_1d(x1 - x2).

    First layer: W_2d[:, 0] = W_1d[:, 0],  W_2d[:, 1] = -W_1d[:, 0]
    All other layers: direct copy (shapes already match).
    """
    tgt_device = next(net_2d.parameters()).device
    sd_1d = torch.load(path_1d, map_location=tgt_device)
    if 'model_state' in sd_1d:
        sd_1d = sd_1d['model_state']
    sd_2d = net_2d.state_dict()

    for key in sd_2d:
        if key not in sd_1d:
            continue
        if sd_1d[key].shape == sd_2d[key].shape:
            sd_2d[key] = sd_1d[key]
        elif key == 'net.0.weight':          # first Linear layer
            w1d = sd_1d[key].squeeze(-1)     # [width]
            sd_2d[key][:, 0] =  w1d
            sd_2d[key][:, 1] = -w1d

    net_2d.load_state_dict(sd_2d)
    print(f"  Loaded 1-D diff checkpoint '{path_1d}' into 2-D net "
          f"(x1-x2 mapping)")
    return net_2d


# ============================================================
# M operator  (randomised impulse, own intervention)
# ============================================================
@torch.no_grad()
def M_lambda_op(phi_net, x_batch, player, lam2, m_samples=2048):
    """
    M_{lam2, player} phi(x1, x2).

    Player 1 shifts x1:
        -lam2 log E_{y~N(0,1)}[ exp(-(phi(y,x2) + l(y-x1)) / lam2) ]
    Player 2 shifts x2 analogously.
    """
    B  = x_batch.size(0)
    x1 = x_batch[:, 0]                                        # [B]
    x2 = x_batch[:, 1]                                        # [B]

    if lam2 >= 0.2:                         # --- soft-min (MC) ---
        y = torch.randn(m_samples, device=device)              # [M]
        y_e = y.unsqueeze(0).expand(B, m_samples)              # [B,M]

        if player == 1:
            fixed = x2.unsqueeze(1).expand(B, m_samples)
            inp   = torch.stack([y_e, fixed], dim=-1)          # [B,M,2]
            jump  = y_e - x1.unsqueeze(1)
        else:
            fixed = x1.unsqueeze(1).expand(B, m_samples)
            inp   = torch.stack([fixed, y_e], dim=-1)
            jump  = y_e - x2.unsqueeze(1)

        phi = phi_net(inp.reshape(-1, 2)).view(B, m_samples)
        ell = l_cost(jump)
        G   = torch.exp(-(phi + ell) / lam2).mean(dim=1)
        return -lam2 * torch.log(G + EPS)

    else:                                   # --- classical min ---
        xi_g = torch.linspace(xi_min, xi_max, m_samples,
                              device=device)                   # [M]
        M = xi_g.size(0)

        if player == 1:
            new  = x1.unsqueeze(1) + xi_g.unsqueeze(0)        # [B,M]
            fix  = x2.unsqueeze(1).expand(B, M)
            inp  = torch.stack([new, fix], dim=-1)
        else:
            fix  = x1.unsqueeze(1).expand(B, M)
            new  = x2.unsqueeze(1) + xi_g.unsqueeze(0)
            inp  = torch.stack([fix, new], dim=-1)

        phi = phi_net(inp.reshape(-1, 2)).view(B, M)
        ell = l_cost(xi_g).unsqueeze(0)                       # [1,M]
        return (phi + ell).min(dim=1).values

# ============================================================
# H operator  (cross-player interaction, classical inf)
# ============================================================
@torch.no_grad()
def H_op(phi_net, x_batch, intervening_player, m_grid=500):
    """
    Cross-player operator with constant cost c.

    H_{21} V^1(x1,x2) = inf_y V^1(x1,y) + c   (player 2 jumps)
    H_{12} V^2(x1,x2) = inf_y V^2(y,x2) + c   (player 1 jumps)
    """
    B  = x_batch.size(0)
    x1 = x_batch[:, 0]
    x2 = x_batch[:, 1]
    yg = torch.linspace(x_min, x_max, m_grid, device=device)  # [G]
    G  = yg.size(0)

    if intervening_player == 2:          # H_{21}: fix x1, sweep x2'
        x1e = x1.unsqueeze(1).expand(B, G)
        ye  = yg.unsqueeze(0).expand(B, G)
        inp = torch.stack([x1e, ye], dim=-1)
    else:                                # H_{12}: fix x2, sweep x1'
        ye  = yg.unsqueeze(0).expand(B, G)
        x2e = x2.unsqueeze(1).expand(B, G)
        inp = torch.stack([ye, x2e], dim=-1)

    phi = phi_net(inp.reshape(-1, 2)).view(B, G)
    return phi.min(dim=1).values + c_cost

# ============================================================
# Trajectory simulation  (pure diffusion, no jumps applied)
# ============================================================
def collect_buffer(n_steps, dt, batch_size):
    """Euler-Maruyama for 2-D state. Returns [(state_t, state_{t+1}), ...]."""
    sqrt_dt = math.sqrt(dt)
    rho = torch.distributions.Uniform(x_min, x_max)
    X = torch.stack([rho.sample((batch_size,)),
                     rho.sample((batch_size,))], dim=-1).to(device)
    buf = []
    for _ in range(n_steps):
        dW = torch.randn_like(X) * sqrt_dt
        X1 = X + b_drift * dt + sigma * dW
        buf.extend(zip(X.cpu(), X1.cpu()))
        X = X1
    return buf

# ============================================================
# One TD gradient step for a single player
# ============================================================
def critic_step(v_net, v_frozen, v_opp_frozen,
                player, lam1, lam2, dt,
                buffer, opt,
                pi_opp_fn=None,
                minibatch=2048):
    """
    Semi-gradient TD update for *player* (1 or 2).

    v_net          – network being trained
    v_frozen       – frozen target  V_bar^{player, n}
    v_opp_frozen   – frozen opponent value net (for opponent pi)
    pi_opp_fn      – if given, callable(x_batch)->pi  (Phase II)
    """
    if len(buffer) < 2:
        return 0.0

    mb  = min(minibatch, len(buffer))
    idx = np.random.choice(len(buffer), mb, replace=False)
    xt, xtp1 = zip(*[buffer[i] for i in idx])
    xt   = torch.stack(xt).to(device)       # [B,2]
    xtp1 = torch.stack(xtp1).to(device)

    # ---- value at current state (keeps gradient) ----
    v_t = v_net(xt)

    with torch.no_grad():
        v_tp1 = v_net(xtp1)

        # M & H for this player (frozen target)
        M_val = M_lambda_op(v_frozen, xt, player, lam2)
        if player == 1:
            H_val = H_op(v_frozen, xt, intervening_player=2)
        else:
            H_val = H_op(v_frozen, xt, intervening_player=1)

        # own pi  (uses M on frozen target, current V detached)
        E_self  = (M_val - v_t.detach()) / lam1
        pi_self = torch.exp(torch.clamp(-E_self, max=25.0)).clamp_max(1.0 / dt)

        # opponent pi
        if pi_opp_fn is not None:
            pi_opp = pi_opp_fn(xt)
        else:
            opp = 2 if player == 1 else 1
            M_opp    = M_lambda_op(v_opp_frozen, xt, opp, lam2)
            v_opp_at = v_opp_frozen(xt)
            E_opp    = (M_opp - v_opp_at) / lam1
            pi_opp   = torch.exp(torch.clamp(-E_opp, max=25.0)).clamp_max(1.0 / dt)

        # survival factor  (product of independent survivals)
        # s = (1 - pi_self*dt) * (1 - pi_opp*dt)
        surv = (1.0 - pi_self * dt).clamp_min(0.0) * (1.0 - pi_opp * dt).clamp_min(0.0)

        # running cost (symmetric, so same for both players)
        f_val = f_run(xt[:, 0], xt[:, 1])

    # TD error  (gradient only through v_t)
    td = (v_t
          - math.exp(-r * dt) * surv * v_tp1
          - dt * (f_val
                  + pi_self * M_val
                  + pi_opp  * H_val
                  - lam1    * R_safe(pi_self)))
    loss = td.square().mean()

    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()

# ============================================================
# Training driver  (Algorithm 1)
# ============================================================
def train_2player_td(lam1, lam2,
                     T=20.0, dt=0.02,
                     roll_batch=4096,
                     N_outer=30,
                     K1_steps=400, K2_steps=400,
                     minibatch=4096, seed=0,
                     init_psi="random",
                     init_1p_path=None):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    n_steps = int(T / dt)

    # --- initialise value networks ---
    v1_net = ValueNet().to(device)
    v2_net = ValueNet().to(device)

    if init_psi == "psi0":
        psi_2d_path = "pinn_models/psi_init_2d_smooth.pt"
        psi_1d_path = "pinn_models/psi_init_smooth.pt"
        if os.path.isfile(psi_2d_path):
            sd = torch.load(psi_2d_path, map_location='cpu')
            v1_net.load_state_dict(sd)
            v2_net.load_state_dict(sd)
            print(f"  Loaded 2D checkpoint '{psi_2d_path}' for both players")
        else:
            print(f"  2D checkpoint not found, falling back to 1D: {psi_1d_path}")
            load_1d_into_2d(v1_net, psi_1d_path, player=1)
            load_1d_into_2d(v2_net, psi_1d_path, player=2)
    elif init_psi == "psi_classical":
        psi_path = "pinn_models/psi_init_classical.pt"
        load_1d_into_2d(v1_net, psi_path, player=1)
        load_1d_into_2d(v2_net, psi_path, player=2)
    elif init_psi == "psi_1p":
        psi_path = init_1p_path or "pinn_models/psi_1p_rl.pt"
        load_1d_diff_into_2d(v1_net, psi_path)
        load_1d_diff_into_2d(v2_net, psi_path)
    else:  # "random"
        print("  Using random initialisation")

    opt1 = optim.AdamW(v1_net.parameters(), lr=1e-3, weight_decay=1e-4)
    opt2 = optim.AdamW(v2_net.parameters(), lr=1e-3, weight_decay=1e-4)

    loss_hist_1, loss_hist_2 = [], []

    for n in range(N_outer):
        print(f"\n{'='*60}")
        print(f"Outer iteration {n}/{N_outer-1}")
        print(f"{'='*60}")

        # Step 7 – freeze targets
        v1_net.eval(); v2_net.eval()
        v1_frozen = deepcopy(v1_net).eval()
        v2_frozen = deepcopy(v2_net).eval()

        # Steps 2-5 – simulate trajectories
        buf = collect_buffer(n_steps, dt, roll_batch)
        print(f"  buffer size = {len(buf)}")

        # ========== Phase I: update Player 1 (Player 2 frozen) ==========
        v1_net.train()
        for k in range(K1_steps):
            l1 = critic_step(v1_net, v1_frozen, v2_frozen,
                             player=1, lam1=lam1, lam2=lam2, dt=dt,
                             buffer=buf, opt=opt1,
                             pi_opp_fn=None, minibatch=minibatch)
            if k % 100 == 0 or k == K1_steps - 1:
                print(f"  [P1  outer {n:2d}  inner {k:3d}]  TD-loss {l1:.3e}")
        loss_hist_1.append(l1)

        # Step 18 – build updated Player-1 policy for Phase II
        v1_updated = deepcopy(v1_net).eval()

        def _pi1_fn(xb, _v1u=v1_updated, _v1f=v1_frozen,
                    _l1=lam1, _l2=lam2, _dt=dt):
            with torch.no_grad():
                Mv = M_lambda_op(_v1f, xb, 1, _l2)
                vv = _v1u(xb)
                E  = (Mv - vv) / _l1
                return torch.exp(torch.clamp(-E, max=25.0)).clamp_max(1.0/_dt)

        # ========== Phase II: update Player 2 (uses updated P1) ==========
        v2_net.train()
        for k in range(K2_steps):
            l2 = critic_step(v2_net, v2_frozen, v1_updated,
                             player=2, lam1=lam1, lam2=lam2, dt=dt,
                             buffer=buf, opt=opt2,
                             pi_opp_fn=_pi1_fn, minibatch=minibatch)
            if k % 100 == 0 or k == K2_steps - 1:
                print(f"  [P2  outer {n:2d}  inner {k:3d}]  TD-loss {l2:.3e}")
        loss_hist_2.append(l2)

        # ---- saves / plots every outer iteration ----
        if True:
            _save_and_plot(v1_net, v2_net, v1_frozen, v2_frozen,
                           lam1, lam2, dt, n, seed,
                           loss_hist_1, loss_hist_2, init_psi,
                           K1_steps, K2_steps)

    # final checkpoint
    torch.save({"v1": v1_net.state_dict(),
                "v2": v2_net.state_dict(),
                "opt1": opt1.state_dict(),
                "opt2": opt2.state_dict(),
                "loss1": loss_hist_1,
                "loss2": loss_hist_2,
                "seed": seed, "init_psi": init_psi,
                "lam1": lam1, "lam2": lam2,
                "K1_steps": K1_steps, "K2_steps": K2_steps},
               f"TD_models_2p/final_lam1_{lam1:.2f}_lam2_{lam2:.3f}"
               f"_initpsi_{init_psi}_seed{seed}.pt")

    with open(f"TD_loss_2p/losses_lam1_{lam1:.2f}_lam2_{lam2:.3f}"
              f"_initpsi_{init_psi}_seed{seed}.pkl", "wb") as fp:
        pickle.dump({"loss1": loss_hist_1, "loss2": loss_hist_2,
                     "K1_steps": K1_steps, "K2_steps": K2_steps}, fp)

    return v1_net.cpu(), v2_net.cpu()

# ============================================================
# Visualisation helpers
# ============================================================
def _save_and_plot(v1, v2, v1f, v2f, lam1, lam2, dt, it, seed,
                   lh1, lh2, init_psi="random",
                   K1_steps=400, K2_steps=400):
    # Evaluate along the anti-diagonal: x1 = u/2, x2 = -u/2
    # so that u = x1 - x2 ranges over [2*x_min, 2*x_max].
    Nu = 300
    u_vals = torch.linspace(2 * x_min, 2 * x_max, Nu, device=device)
    x1_line = u_vals / 2.0
    x2_line = -u_vals / 2.0
    line_pts = torch.stack([x1_line, x2_line], dim=-1)   # [Nu, 2]

    with torch.no_grad():
        V1_line = v1(line_pts).cpu().numpy()
        V2_line = v2(line_pts).cpu().numpy()

        # policies
        M1v = M_lambda_op(v1f, line_pts, 1, lam2)
        v1v = v1(line_pts)
        pi1_line = torch.exp(torch.clamp(-(M1v - v1v) / lam1, max=25.0)
                             ).clamp_max(1.0 / dt).cpu().numpy()

        M2v = M_lambda_op(v2f, line_pts, 2, lam2)
        v2v = v2(line_pts)
        pi2_line = torch.exp(torch.clamp(-(M2v - v2v) / lam1, max=25.0)
                             ).clamp_max(1.0 / dt).cpu().numpy()

    u_np = u_vals.cpu().numpy()
    tag = f"vRL_lam1_{lam1:.2f}_lam2_{lam2:.3f}_initpsi_{init_psi}_iter_{it}_seed_{seed}"

    # --- value functions ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(u_np, V1_line, 'b-', lw=2, label=r'$V^1$')
    ax.plot(u_np, V2_line, 'r-', lw=2, label=r'$V^2$')
    ax.set_xlabel(r'$x_1 - x_2$'); ax.set_ylabel('Value')
    ax.set_title(f'Value functions  iter {it}  '
                 rf'$\lambda_1$={lam1} $\lambda_2$={lam2}  init={init_psi}  seed={seed}')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"TD_figures_2p/values_{tag}.png", dpi=150)
    plt.close("all")

    # --- policies ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(u_np, pi1_line, 'b-', lw=2, label=r'$\pi_1$')
    ax.plot(u_np, pi2_line, 'r-', lw=2, label=r'$\pi_2$')
    ax.set_xlabel(r'$x_1 - x_2$'); ax.set_ylabel('Intensity')
    ax.set_title(f'Intervention intensities  iter {it}  '
                 rf'$\lambda_1$={lam1} $\lambda_2$={lam2}  init={init_psi}  seed={seed}')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"TD_figures_2p/policies_{tag}.png", dpi=150)
    plt.close("all")

    # --- loss curves ---
    if len(lh1) > 1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(lh1, label='Player 1')
        ax.semilogy(lh2, label='Player 2')
        ax.set_xlabel('Outer iteration'); ax.set_ylabel('TD loss')
        ax.legend(); ax.grid(True)
        fig.suptitle(f'TD loss  '
                     rf'$\lambda_1$={lam1} $\lambda_2$={lam2}  init={init_psi}  seed={seed}')
        plt.tight_layout()
        plt.savefig(f"TD_figures_2p/loss_{tag}.png", dpi=150)
        plt.close("all")

    # save model checkpoint
    torch.save({"v1": v1.state_dict(), "v2": v2.state_dict(),
                "seed": seed, "init_psi": init_psi,
                "lam1": lam1, "lam2": lam2, "iter": it,
                "K1_steps": K1_steps, "K2_steps": K2_steps},
               f"TD_models_2p/ckpt_{tag}.pt")

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    Path("TD_models_2p").mkdir(exist_ok=True)
    Path("TD_figures_2p").mkdir(exist_ok=True)
    Path("TD_loss_2p").mkdir(exist_ok=True)

    ap = argparse.ArgumentParser(
        description="2-Player Randomised Impulse Control — TD algorithm")
    ap.add_argument("--init_psi",   type=str,   default="psi0",
                    choices=["random", "psi0", "psi_classical", "psi_1p"],
                    help="initialisation: random | psi0 | psi_classical | psi_1p")
    ap.add_argument("--init_1p_path", type=str, default=None,
                    help="path to 1-D checkpoint for psi_1p init "
                         "(default: pinn_models/psi_1p_rl.pt)")
    ap.add_argument("--lam1",       type=float, default=1.0)
    ap.add_argument("--lam2",       type=float, default=1.0)
    ap.add_argument("--N_outer",    type=int,   default=30)
    ap.add_argument("--K1_steps",   type=int,   default=400)
    ap.add_argument("--K2_steps",   type=int,   default=400)
    ap.add_argument("--roll_batch", type=int,   default=4096)
    ap.add_argument("--T",          type=float, default=20.0)
    ap.add_argument("--dt",         type=float, default=0.02)
    ap.add_argument("--minibatch",  type=int,   default=4096)
    ap.add_argument("--seed",       type=int,   default=0)
    args = ap.parse_args()

    print(f"\n=== 2-Player Game: "
          f"lam1={args.lam1:.2f}  lam2={args.lam2:.3f} ===")
    print(f"Device: {device}")

    train_2player_td(
        lam1=args.lam1,       lam2=args.lam2,
        T=args.T,             dt=args.dt,
        roll_batch=args.roll_batch,
        N_outer=args.N_outer,
        K1_steps=args.K1_steps, K2_steps=args.K2_steps,
        minibatch=args.minibatch, seed=args.seed,
        init_psi=args.init_psi,
        init_1p_path=args.init_1p_path)
