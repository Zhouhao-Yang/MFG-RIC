"""
train_pinn_2d_init.py
Train initial value-function approximations for RL_RIC_TD*.py.

Modes
-----
2d_smooth   – 2-D ValueNet fitting V^0(x1,x2) = 2 exp(-|x1-x2|) + 2|x1-x2|
              via PINN loss.  Saves  pinn_models/psi_init_2d_smooth.pt
1d_classical – 1-D PsiNet fitting the classical impulse-control solution
              loaded from classical_V.pkl.  Saves  pinn_models/psi_init_classical.pt
"""

import math
import pickle
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# Parameters (must match RL_RIC_TD_2player.py)
# ============================================================
r = 0.5
sigma = math.sqrt(2) / 2       # => sigma^2 / 2 = 1/4
x_min, x_max = -3.0, 3.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================
# Analytical solution
# ============================================================
def V0_analytical(x1, x2):
    """V^0(x1, x2) = 2 exp(-|x1-x2|) + 2|x1-x2|."""
    d = torch.abs(x1 - x2)
    return 2.0 * torch.exp(-d) + 2.0 * d


# ============================================================
# Network (same architecture as in RL_RIC_TD_2player.py)
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
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)


# ============================================================
# 1-D PsiNet (same architecture as RL_RIC_TD.py)
# ============================================================
class PsiNet(nn.Module):
    def __init__(self, width=128, depth=4):
        super().__init__()
        layers, d_in = [], 1
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
        return self.net(x.unsqueeze(-1)).squeeze(-1)


# ============================================================
# Classical impulse-control solution (loaded from classical_V.pkl)
# ============================================================
def build_classical_V(pkl_path="classical_V.pkl"):
    """Return a vectorised V(x) callable from the classical solution."""
    with open(pkl_path, "rb") as f:
        c1, c2, d, D, u, U = pickle.load(f)
    print(f"  Classical solution: d={d:.3f} D={D:.3f} U={U:.3f} u={u:.3f} "
          f"c1={c1:.6f} c2={c2:.6f}")

    # Parameters must match the ones used to solve classical_V.pkl
    _r, _mu, _sigma = 0.50, 0.00, 1.00
    _h, _p = 1.0, 1.0
    _kp = 1.0
    _km = 1.0

    _disc = _mu**2 + 2*_r*_sigma**2
    _t1 = (-_mu + np.sqrt(_disc)) / _sigma**2
    _t2 = (-_mu - np.sqrt(_disc)) / _sigma**2

    _A = (_h + _p) / _r
    _C1 = c1 - _A * _t2 / (_t1 * (_t1 - _t2))
    _C2 = c2 + _A * _t1 / (_t2 * (_t1 - _t2))

    def V(x):
        """Evaluate classical value at scalar or array x."""
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x)
        # Region 1: x <= d  (linear extension)
        m1 = x <= d
        Vd = -_p*d/_r + _C1*np.exp(_t1*d) + _C2*np.exp(_t2*d)
        out[m1] = Vd + _kp*(d - x[m1])
        # Region 2: d < x <= 0
        m2 = (x > d) & (x <= 0)
        out[m2] = -_p*x[m2]/_r + _C1*np.exp(_t1*x[m2]) + _C2*np.exp(_t2*x[m2])
        # Region 3: 0 < x <= u
        m3 = (x > 0) & (x <= u)
        out[m3] = _h*x[m3]/_r + c1*np.exp(_t1*x[m3]) + c2*np.exp(_t2*x[m3])
        # Region 4: x > u  (linear extension)
        m4 = x > u
        Vu = _h*u/_r + c1*np.exp(_t1*u) + c2*np.exp(_t2*u)
        out[m4] = Vu + _km*(x[m4] - u)
        return out

    return V


# ============================================================
# Training: 1-D classical
# ============================================================
def train_1d_classical(epochs=8000, lr=1e-3, pkl_path="classical_V.pkl"):
    """Fit a PsiNet to the classical solution on [-3, 3]."""
    V_classical = build_classical_V(pkl_path)

    x_np = np.linspace(x_min, x_max, 3001)
    y_np = V_classical(x_np).astype(np.float32)

    x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_np, dtype=torch.float32, device=device)

    net = PsiNet().to(device)
    opt = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)

    print(f"Training 1D PsiNet (classical) on {device}")
    print(f"  Grid points: {len(x_np)},  Epochs: {epochs}\n")

    best_loss, best_sd = float("inf"), None

    for ep in range(epochs + 1):
        pred = net(x_t)
        loss = (pred - y_t).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_sd = {k: v.clone() for k, v in net.state_dict().items()}

        if ep % 1000 == 0 or ep == epochs:
            print(f"  epoch {ep:5d}/{epochs}  "
                  f"MSE = {loss.item():.6e}  "
                  f"lr = {opt.param_groups[0]['lr']:.1e}")

    net.load_state_dict(best_sd)
    net.eval()
    print(f"\n  Best MSE = {best_loss:.6e}")

    # Quick plot
    with torch.no_grad():
        pred_np = net(x_t).cpu().numpy()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_np, y_np, "b-", lw=2, label="Classical V(x)")
    ax.plot(x_np, pred_np, "r--", lw=2, label="PsiNet fit")
    ax.set_xlabel("x"); ax.set_ylabel("V")
    ax.set_title("1D Classical Fit"); ax.legend(); ax.grid(True)
    plt.tight_layout()
    save_dir = Path("pinn_models")
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "psi_classical_fit.png", dpi=150)
    plt.close()

    return net, best_loss


# ============================================================
# PDE residual via autograd
# ============================================================
def pde_residual(net, x):
    """
    Compute PDE residual:  (1/4)(V_{x1x1} + V_{x2x2}) - r*V + |x1-x2|

    x: [B, 2] with requires_grad=True
    Returns: [B] residual values
    """
    V = net(x)                                    # [B]
    # First derivatives
    dV = torch.autograd.grad(V.sum(), x, create_graph=True)[0]  # [B, 2]
    Vx1 = dV[:, 0]
    Vx2 = dV[:, 1]
    # Second derivatives
    Vx1x1 = torch.autograd.grad(Vx1.sum(), x, create_graph=True)[0][:, 0]
    Vx2x2 = torch.autograd.grad(Vx2.sum(), x, create_graph=True)[0][:, 1]

    f = torch.abs(x[:, 0] - x[:, 1])             # running cost |x1-x2|
    residual = 0.25 * (Vx1x1 + Vx2x2) - r * V + f
    return residual


# ============================================================
# Training
# ============================================================
def train(epochs=2000, n_data=4096, n_colloc=4096,
          w_data=1.0, w_pde=0.1, lr=1e-3):

    net = ValueNet().to(device)

    # --- Data points (dense grid + random) ---
    Ng = 64
    g1 = torch.linspace(x_min, x_max, Ng, device=device)
    G1, G2 = torch.meshgrid(g1, g1, indexing='ij')
    x_grid = torch.stack([G1.reshape(-1), G2.reshape(-1)], dim=-1)  # [Ng^2, 2]
    y_grid = V0_analytical(x_grid[:, 0], x_grid[:, 1])

    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                     eta_min=1e-6)

    best_loss = float('inf')
    best_sd = None

    print(f"Training 2D PINN on {device}")
    print(f"  Grid data points:  {x_grid.shape[0]}")
    print(f"  Collocation points per epoch: {n_colloc}")
    print(f"  Weights: w_data={w_data}, w_pde={w_pde}")
    print(f"  Epochs: {epochs}\n")

    for ep in range(epochs):
        net.train()

        # --- Data loss (grid) ---
        pred_grid = net(x_grid)
        data_loss = (pred_grid - y_grid).square().mean()

        # --- Data loss (random points) ---
        x_rand = (x_min + (x_max - x_min)
                  * torch.rand(n_data, 2, device=device))
        y_rand = V0_analytical(x_rand[:, 0], x_rand[:, 1])
        pred_rand = net(x_rand)
        data_loss = data_loss + (pred_rand - y_rand).square().mean()

        # --- PDE residual loss (collocation) ---
        x_col = (x_min + (x_max - x_min)
                 * torch.rand(n_colloc, 2, device=device))
        x_col.requires_grad_(True)
        res = pde_residual(net, x_col)
        pde_loss = res.square().mean()

        loss = w_data * data_loss + w_pde * pde_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_sd = {k: v.clone() for k, v in net.state_dict().items()}

        if ep % 200 == 0 or ep == epochs - 1:
            print(f"  [Epoch {ep:4d}/{epochs}]  "
                  f"total={loss.item():.3e}  "
                  f"data={data_loss.item():.3e}  "
                  f"pde={pde_loss.item():.3e}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    # Restore best
    net.load_state_dict(best_sd)
    net.eval()

    # --- Final metrics ---
    with torch.no_grad():
        pred_final = net(x_grid)
        mse_final = (pred_final - y_grid).square().mean().item()
    x_test = (x_min + (x_max - x_min)
              * torch.rand(8192, 2, device=device))
    x_test.requires_grad_(True)
    with torch.no_grad():
        y_test = V0_analytical(x_test[:, 0], x_test[:, 1])
        pred_test = net(x_test)
        mse_test = (pred_test - y_test).square().mean().item()
    res_test = pde_residual(net, x_test)
    pde_final = res_test.square().mean().item()

    print(f"\n{'='*50}")
    print(f"Final grid MSE:     {mse_final:.3e}")
    print(f"Final random MSE:   {mse_test:.3e}")
    print(f"Final PDE residual: {pde_final:.3e}")
    print(f"{'='*50}")

    return net, mse_final, pde_final


# ============================================================
# Plotting
# ============================================================
def plot_comparison(net, save_dir):
    Ng = 100
    g = torch.linspace(x_min, x_max, Ng, device=device)
    G1, G2 = torch.meshgrid(g, g, indexing='ij')
    grid = torch.stack([G1.reshape(-1), G2.reshape(-1)], dim=-1)

    with torch.no_grad():
        V_net = net(grid).view(Ng, Ng).cpu().numpy()
    V_ana = V0_analytical(G1, G2).view(Ng, Ng).cpu().numpy()
    G1n, G2n = G1.cpu().numpy(), G2.cpu().numpy()

    # --- 2D contour comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    c0 = axes[0].contourf(G1n, G2n, V_ana, levels=50, cmap='viridis')
    axes[0].set_title('Analytical $V^0(x_1,x_2)$')
    axes[0].set_xlabel('$x_1$'); axes[0].set_ylabel('$x_2$')
    plt.colorbar(c0, ax=axes[0])

    c1 = axes[1].contourf(G1n, G2n, V_net, levels=50, cmap='viridis')
    axes[1].set_title('PINN $V^0(x_1,x_2)$')
    axes[1].set_xlabel('$x_1$'); axes[1].set_ylabel('$x_2$')
    plt.colorbar(c1, ax=axes[1])

    err = np.abs(V_net - V_ana)
    c2 = axes[2].contourf(G1n, G2n, err, levels=50, cmap='hot')
    axes[2].set_title('|Error|')
    axes[2].set_xlabel('$x_1$'); axes[2].set_ylabel('$x_2$')
    plt.colorbar(c2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_dir / "pinn_2d_contour.png", dpi=150)
    plt.close()

    # --- Diagonal slices ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Slice 1: x2 = 0 (vary x1)
    x1_line = torch.linspace(x_min, x_max, 200, device=device)
    x2_zero = torch.zeros_like(x1_line)
    pts = torch.stack([x1_line, x2_zero], dim=-1)
    with torch.no_grad():
        v_nn = net(pts).cpu().numpy()
    v_an = V0_analytical(x1_line, x2_zero).cpu().numpy()
    x1_np = x1_line.cpu().numpy()

    axes[0].plot(x1_np, v_an, 'b-', lw=2, label='Analytical')
    axes[0].plot(x1_np, v_nn, 'r--', lw=2, label='PINN')
    axes[0].set_xlabel('$x_1$'); axes[0].set_ylabel('$V^0$')
    axes[0].set_title('Slice: $x_2 = 0$')
    axes[0].legend(); axes[0].grid(True)

    # Slice 2: x1 = x2 (diagonal)
    diag = torch.linspace(x_min, x_max, 200, device=device)
    pts_diag = torch.stack([diag, diag], dim=-1)
    with torch.no_grad():
        v_nn_d = net(pts_diag).cpu().numpy()
    v_an_d = V0_analytical(diag, diag).cpu().numpy()
    diag_np = diag.cpu().numpy()

    axes[1].plot(diag_np, v_an_d, 'b-', lw=2, label='Analytical')
    axes[1].plot(diag_np, v_nn_d, 'r--', lw=2, label='PINN')
    axes[1].set_xlabel('$x_1 = x_2$'); axes[1].set_ylabel('$V^0$')
    axes[1].set_title('Slice: $x_1 = x_2$ (diagonal)')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_dir / "pinn_2d_slices.png", dpi=150)
    plt.close()

    print(f"Plots saved to {save_dir}/")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train initial value-function approximations")
    ap.add_argument("--mode", type=str, default="2d_smooth",
                    choices=["2d_smooth", "1d_classical"],
                    help="2d_smooth: 2D PINN V^0 | 1d_classical: 1D classical fit")
    ap.add_argument("--epochs",   type=int,   default=None,
                    help="training epochs (default: 2000 for 2d, 8000 for 1d)")
    ap.add_argument("--n_data",   type=int,   default=4096)
    ap.add_argument("--n_colloc", type=int,   default=4096)
    ap.add_argument("--w_data",   type=float, default=1.0)
    ap.add_argument("--w_pde",    type=float, default=0.1)
    ap.add_argument("--lr",       type=float, default=1e-3)
    args = ap.parse_args()

    save_dir = Path("pinn_models")
    save_dir.mkdir(exist_ok=True)

    if args.mode == "1d_classical":
        epochs = args.epochs if args.epochs is not None else 8000
        net, mse = train_1d_classical(epochs=epochs, lr=args.lr)
        out_path = save_dir / "psi_init_classical.pt"
        torch.save(net.cpu().state_dict(), out_path)
        print(f"\nCheckpoint saved to {out_path}")

    else:  # 2d_smooth
        epochs = args.epochs if args.epochs is not None else 2000
        net, mse, pde_res = train(
            epochs=epochs, n_data=args.n_data, n_colloc=args.n_colloc,
            w_data=args.w_data, w_pde=args.w_pde, lr=args.lr)
        out_path = save_dir / "psi_init_2d_smooth.pt"
        torch.save(net.state_dict(), out_path)
        print(f"\nCheckpoint saved to {out_path}")
        plot_comparison(net, save_dir)
