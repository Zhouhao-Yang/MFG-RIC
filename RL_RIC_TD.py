from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import pickle
from scipy.stats import norm, multivariate_normal
from scipy.integrate import solve_bvp
import warnings
#from tqdm.auto import tqdm  # nice progress bars
from pathlib import Path
from scipy.optimize import newton_krylov
from scipy.sparse import diags
import torch, torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import math, time, os, sys
import argparse

params = {
    "r":        0.50,
    "mu":       0.00,
    "sigma":    1.00,
    "h":        1.0,
    "p":        1.0,
    "Kp":       3.0,
    "kp":       1.0,
    "Km":       3.0,
    "km":       1.0,
    "x_min":   -3.0,
    "x_max":    3.0,
    "Nx":      3001,
    "Nxi":    3001,
    "xi_min":  -3.0,
    "xi_max":   3.0,
    "sampler": {
        "n_walkers": 8,
        "burn_steps": 500,
        "prod_steps": 1250,
        "xi_std":     4.0
    },
    "n_jobs":   4,       # for parallel N_lambda
    "tol_psi":  1e-3,
    "max_iter": 50
}

# Unpack
(r, mu, sigma, h, p, Kp, kp, Km, km,
 x_min, x_max, Nx, Nxi,
 xi_min, xi_max,
 sampler_cfg,
 n_jobs, tol_psi, max_iter) = (
    params["r"], params["mu"], params["sigma"],
    params["h"], params["p"], params["Kp"], params["kp"],
    params["Km"], params["km"],
    params["x_min"], params["x_max"], params["Nx"], params["Nxi"],
    params["xi_min"], params["xi_max"],
    params["sampler"],
    params["n_jobs"], params["tol_psi"], params["max_iter"]
)

def save_params(params, filename='params.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(params, f)
# Load the parameters from a pickle file
def load_params(filename='params.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)
x_grid = np.linspace(x_min, x_max, Nx)
print(f"x_grid: {x_grid}")
dx = x_grid[1] - x_grid[0]
xi_grid = np.linspace(xi_min, xi_max, Nxi)
dxi = xi_grid[1] - xi_grid[0]

(c1,c2,d,D,u,U) = load_params('classical_V.pkl')

disc = mu**2 + 2*r*sigma**2
t1 = (-mu + np.sqrt(disc))/sigma**2      # >1
t2 = (-mu - np.sqrt(disc))/sigma**2      # <0
def V_pos(x,c1,c2):
    return  h*x/r + mu*h/r**2 + c1*np.exp(t1*x)+c2*np.exp(t2*x)
def V_pos_p(x,c1,c2):
    return  h/r + c1*t1*np.exp(t1*x)+c2*t2*np.exp(t2*x)

def coeff_neg(c1,c2):
    A=(h+p)/r; B=(h+p)*mu/r**2
    C1 = c1 - A*t2/(t1*(t1-t2)) 
    C2 = c2 + A*t1/(t2*(t1-t2)) 
    return C1,C2
def V_neg(x,c1,c2):
    C1, C2 = coeff_neg(c1,c2)
    return -p*x/r - mu*p/r**2 + C1*np.exp(t1*x)+C2*np.exp(t2*x)
def V_neg_p(x,c1,c2):
    C1, C2 = coeff_neg(c1,c2)
    return -p/r + C1*t1*np.exp(t1*x)+C2*t2*np.exp(t2*x)

def V(x):

    if x<=d:   return V_neg(d,c1,c2)+kp*(d-x)
    if x<=0:   return V_neg(x,c1,c2)
    if x<=u:   return V_pos(x,c1,c2)
    return V_pos(u,c1,c2)+km*(x-u)

classical_psi = [V(x) for x in x_grid]


EPS = 1e-12


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

r, mu, sigma = 0.50, 0.00, 1.00
h_coef, p_coef = 1.0, 1.0
Kp, kp, Km, km = 3.0, 1.0, 3.0, 1.0
EPS = 1e-12                                      # global tiny number

# costs -----------------------------------------------------------------------
def f_run(x):   return torch.where(x >= 0,  h_coef*x, -p_coef*x)
def l_cost(z):  return torch.where(z >= 0, Kp + kp*z, Km - km*z)
def R_safe(pi): return pi - pi.clamp_min(EPS)*torch.log(pi.clamp_min(EPS))


# critic net ------------------------------------------------------------------
class PsiNet(nn.Module):
    def __init__(self, width=128, depth=4):
        super().__init__()
        layers, d = [], 1
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x.unsqueeze(-1)).squeeze(-1)

# non‑local operator ----------------------------------------------------------
@torch.no_grad()
def N_lambda_mc_vec(phi_net, x_batch, lam2, m_samples=4096):

    if lam2 >= 0.2:
        B, M = x_batch.size(0), m_samples
        xi   = torch.randn(B, M, device=device)            # ξ ~ N(0,1)
        Z    = -x_batch.unsqueeze(1) + xi                  # Z = -x+ξ
        ell  = l_cost(Z)
        phi  = phi_net(xi.reshape(-1)).view(B, M)
        G    = torch.exp(-(phi + ell) / lam2).mean(dim=1)
        return -lam2 * torch.log(G + EPS)
    else:
        B, M = x_batch.size(0), m_samples
        # use classical nonlocal operator N\phi(x) = inf_\xi (\phi(x+\xi) + l(\xi))
        xi_grid = torch.linspace(xi_min, xi_max, m_samples, device=device)   # (M,)
        M       = xi_grid.size(0)

        # Evaluate φ(x+ξ) for every ξ on the grid in one vectorised shot
        #   shapes:  X      = (B,1)
        #            XI     = (1,M)
        #            X+XI   = (B,M)
        Z   = x_batch.unsqueeze(1) + xi_grid.unsqueeze(0)                    # (B,M)
        phi = phi_net(Z.reshape(-1)).view(B, M)                              # (B,M)

        # ℓ depends on ξ only ⇒ pre‑compute once
        ell = l_cost(xi_grid).unsqueeze(0)                                   # (1,M)

        # point‑wise min over the ξ‑axis
        N_vals = (phi + ell).min(dim=1).values                               # (B,)

        return N_vals


# ------------------------------------------------------------------
# 1.  rollout  (fresh buffer each outer loop)   ★ no early return
# ------------------------------------------------------------------
def collect_buffer(psi_net, lam1, lam2, rho,
                   n_steps, dt, batch_size, p_min):
    buffer, sqrt_dt = [], math.sqrt(dt)
    EXP_CLIP, PI_MAX = 15.0, 1.0/dt
    X = rho.sample((batch_size,)).to(device)

    for _ in range(n_steps):
        # with torch.no_grad():
        #     Nvals = N_lambda_mc_vec(psi_net, X, lam2)
        #     E     = (Nvals - psi_net(X)) / lam1
        #     pi    = torch.exp(torch.clamp(-E, max=EXP_CLIP)).clamp_max(PI_MAX)
        dW  = torch.randn_like(X) * sqrt_dt
        X1  = X + mu*dt + sigma*dW
        buffer.extend(zip(X.cpu(), X1.cpu()))
        # live = torch.rand_like(pi) > p_min
        # if not live.any(): break
        # X = X[live]
    return buffer                                # ★ outside the loop

# ------------------------------------------------------------------
# 2.  one TD step  (re‑uses ADAM optimizer)      ★ no re‑init of opt
# ------------------------------------------------------------------
def critic_step(psi_net, psi_eval, lam1, lam2,
                dt, buffer, opt,
                minibatch=2048):
    if len(buffer) < 2: return 0.0

    mb = min(minibatch, len(buffer))
    idx = np.random.choice(len(buffer), mb, replace=False)
    x_t, x_tp1 = zip(*[buffer[i] for i in idx])
    x_t   = torch.stack(x_t).to(device)
    x_tp1 = torch.stack(x_tp1).to(device)

    psi_t = psi_net(x_t)

    with torch.no_grad():
        psi_tp1 = psi_net(x_tp1)
        N_t     = N_lambda_mc_vec(psi_eval, x_t, lam2)     # frozen ψⁿ
        E_t     = (N_t - psi_t.detach()) / lam1
        pi_t    = torch.exp(torch.clamp(-E_t, max=25.0)).clamp_max(1.0/dt)

    td = psi_t - math.exp(-r*dt)*psi_tp1* (1-pi_t*dt) \
         - dt*( f_run(x_t) + pi_t*N_t - lam1*R_safe(pi_t) )
    loss = td.square().mean()

    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


# ------------------------------------------------------------------
# 3.  training driver  (outer FP loop)          ★ buffer renewed
# ------------------------------------------------------------------
def train_randomised_td(lam1, lam2, initial_psi="psi0",
                        T=20.0, dt=0.02,
                        roll_batch=4096,
                        N_outer=30, gd_steps=400,
                        minibatch=4096, seed=0):

    np.random.seed(seed)            # NumPy rng
    torch.manual_seed(seed)         # CPU  rng
    torch.cuda.manual_seed_all(seed)  # GPU  rng (safe even on CPU–only)
    n_steps  = int(T/dt)
    rho      = torch.distributions.Uniform(x_min, x_max)
    p_min    = 1e-3

    Outer_relative_l2_loss = []
    xx       = torch.linspace(x_min, x_max, Nx, device=device)

    psi_classical_net = PsiNet().to(device)
    psi_classical_net.load_state_dict(torch.load('pinn_models/psi_init_classical.pt', map_location=device))
    classical_psi = psi_classical_net(xx).cpu().detach().numpy()

    psi_net  = PsiNet().to(device)
    if initial_psi == "random":
        pass
    elif initial_psi == "psi0":
        psi_net.load_state_dict(torch.load('pinn_models/psi_init_smooth.pt', map_location=device))
    elif initial_psi == "psi_classical":
        psi_net.load_state_dict(torch.load('pinn_models/psi_init_classical.pt', map_location=device))

    opt      = optim.AdamW(psi_net.parameters(), lr=1e-3, weight_decay=1e-4)


    for n in range(N_outer):

        # ----- freeze critic, collect new buffer under πⁿ ------------
        psi_net.eval()
        #compute relative l2 loss
        with torch.no_grad():
            relative_l2_loss = torch.norm(psi_net(xx) - psi_classical_net(xx)) / torch.norm(psi_classical_net(xx))
            Outer_relative_l2_loss.append(relative_l2_loss.item())
            print(f"Outer iteration {n}, relative l2 loss: {relative_l2_loss.item():.3e}")


        buffer = collect_buffer(psi_net, lam1, lam2, rho,
                                n_steps, dt, roll_batch, p_min)

        # ----- inner critic updates ---------------------------------
        psi_net.train()
        psi_eval = deepcopy(psi_net).eval()          # frozen ψⁿ
        for k in range(gd_steps):
            loss = critic_step(psi_net, psi_eval,
                               lam1, lam2, dt, buffer, opt, minibatch)
            if k % 100 == 0 or k == gd_steps-1:
                print(f"[outer {n:2d}  inner {k:3d}]  "
                      f"TD‑loss {loss:.3e}  buffer={len(buffer)}")
    
        if n % 1 == 0 or n == N_outer-1:
            with torch.no_grad():
                plt.figure(figsize=(8,4))
                plt.plot(xx.cpu(), psi_net(xx).cpu().detach().numpy(), label=rf"$\psi^*$, λ₁={lam1}")
                plt.plot(xx.cpu(), classical_psi, '--', label=r'Classical $\psi$')
                plt.grid(); plt.legend(); plt.title(f"TD-error solutions, lam1=lam2= {lam1}, init_psi={initial_psi}, Iter {n}, Seed {seed}")
                #save the figure
                plt.savefig(f"TD_figures/psi_lam1_{lam1:.2f}_initpsi_{initial_psi}_iter_{n}_sigma_{sigma}_seed_{seed}.png")
                plt.close("all")
            torch.save({"model_state": psi_net.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "seed": seed,
                        "init_psi": initial_psi,
                        "lam1": lam1, "lam2": lam2,
                        "iter": n, "sigma": sigma},
                       f"TD_models/psi_net_lam1_{lam1:.2f}_lam2_{lam2:.3f}_initpsi_{initial_psi}_iter_{n}_sigma_{sigma}_seed{seed}.pt")
    
    # save the relative l2 loss
    with open(f"TD_loss/Outer_relative_l2_loss_lam1_{lam1:.2f}_lam2_{lam2:.3f}_initpsi_{initial_psi}_sigma_{sigma}_seed{seed}.pkl", "wb") as f:
        pickle.dump(Outer_relative_l2_loss, f)

    # save raw state_dict for 2-player initialisation
    Path("pinn_models").mkdir(exist_ok=True)
    torch.save(psi_net.state_dict(), "pinn_models/psi_1p_rl.pt")
    print("Saved single-player RL checkpoint → pinn_models/psi_1p_rl.pt")

    return psi_net.cpu()

# ------------------------------------------------------------------
# 4.  run grid
# ------------------------------------------------------------------
if __name__ == "__main__":
    Path("TD_models").mkdir(exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_psi", type=str, default="classical", help="initial_psi")
    parser.add_argument("--lam1", type=float, default=1.0, help="lambda 1")
    parser.add_argument("--lam2", type=float, default=1.0, help="lambda 2")
    parser.add_argument("--N_outer", type=int, default=30, help="number of outer iterations")
    parser.add_argument("--gd_steps", type=int, default=400, help="number of gradient descent steps")
    parser.add_argument("--roll_batch", type=int, default=4096, help="batch size for rollout")
    parser.add_argument("--T", type=float, default=20.0, help="total time for rollout")
    parser.add_argument("--dt", type=float, default=0.02, help="time step for rollout")
    parser.add_argument("--minibatch", type=int, default=4096, help="minibatch size for critic step")
    parser.add_argument("--seed", type=int, default=0, help="random seed for reproducibility")
    args = parser.parse_args()
    initial_psi = args.init_psi 
    lam1 = args.lam1
    lam2 = args.lam2
    N_outer = args.N_outer
    gd_steps = args.gd_steps
    roll_batch = args.roll_batch
    T = args.T
    dt = args.dt
    minibatch = args.minibatch
    seed = args.seed
    print(f"Training with initial_psi={initial_psi}")


    print(f"\n=== λ₁={lam1:.2f}  λ₂={lam2:.3f} ===")
    net = train_randomised_td(lam1, lam2, initial_psi,T=T, dt=dt,roll_batch=roll_batch,N_outer=N_outer, gd_steps=gd_steps,minibatch=minibatch,seed=seed)
    torch.save({"model_state": net.state_dict(),
                "seed": seed, "init_psi": initial_psi,
                "lam1": lam1, "lam2": lam2, "sigma": sigma},
            f"TD_models/psi_net_lam1_{lam1:.2f}_lam2_{lam2:.3f}_initpsi_{initial_psi}_sigma_{sigma}_seed_{seed}.pt")
