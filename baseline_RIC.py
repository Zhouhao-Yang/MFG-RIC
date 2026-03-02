from scipy.optimize import fsolve
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# ----- parameters from your file -----
r    = 0.10
mu   = 0.03   # this is 'b' in the derivation
sigma= 0.20
h    = 1.0
p    = 1.0
Kp   = 2.0
kp   = 0.5
Km   = 2.0
km   = 0.5

disc = mu**2 + 2*r*sigma**2
t1 = (-mu + np.sqrt(disc))/sigma**2      # >1
t2 = (-mu - np.sqrt(disc))/sigma**2      # <0
print(f"t1 = {t1:.4f},   t2 = {t2:.4f}")


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
    C1,C2 = coeff_neg(c1,c2)
    return -p*x/r - mu*p/r**2 + C1*np.exp(t1*x)+C2*np.exp(t2*x)
def V_neg_p(x,c1,c2):
    C1,C2 = coeff_neg(c1,c2)
    return -p/r + C1*t1*np.exp(t1*x)+C2*t2*np.exp(t2*x)

# six equations with correct signs
def F(X):
    c1,c2,d,D,u,U = X
    return [
        V_neg_p(d,c1,c2)  + kp,                     # V'(d) = -k⁺
        V_neg_p(D,c1,c2)  + kp,                     # V'(D) = -k⁺
        V_neg(d,c1,c2) - (V_neg(D,c1,c2)+Kp+kp*(D-d)),
        V_pos_p(u,c1,c2) - km,                     # V'(u) =  k⁻
        V_pos_p(U,c1,c2) - km,                     # V'(U) =  k⁻
        V_pos(u,c1,c2) - (V_pos(U,c1,c2)+Km+km*(u-U))
    ]

sol = fsolve(F,[-1,1,-1,-0.1,1,0.1])  # The initial guess is IMPORTANT! I think the problem is that the linear system is ill-conditioned.
c1,c2,d,D,u,U = sol
print(f"d={d:.3f}, D={D:.3f}, u={u:.3f}, U={U:.3f}, c1={c1:.3f}, c2={c2:.3f}")

def V(x):
    if x<=d:   return V_neg(d,c1,c2)+kp*(d-x)
    if x<=0:   return V_neg(x,c1,c2)
    if x<=u:   return V_pos(x,c1,c2)
    return V_pos(u,c1,c2)+km*(x-u)


# ======================
# Classical QVI baseline
# ======================

# grid just on the window you plot
x_min_num, x_max_num, Nx_num = -3, 3, 3001
x_grid = np.linspace(x_min_num, x_max_num, Nx_num)
dx = x_grid[1] - x_grid[0]

def estimate_drift_and_diffusion_from_buffer(buffer, dt):
    """
    Classical model-based baseline: estimate drift b and volatility sigma
    of the uncontrolled diffusion from one-step increments in `buffer`.

    Parameters
    ----------
    buffer : list of (x_t, x_tp1) tensors on CPU
             (as returned by collect_buffer)
    dt     : float
        Time step between observations.

    Returns
    -------
    b_hat : float   # drift estimate
    sigma_hat : float  # volatility estimate
    """
    if len(buffer) == 0:
        raise ValueError("Buffer is empty; cannot estimate parameters.")

    # One-step increments ΔX_i = X_{t_{i+1}} - X_{t_i}
    dX = np.array([(x1.item() - x0.item()) for (x0, x1) in buffer])
    N  = dX.shape[0]

    # Reviewer’s estimators:
    #   σ^2 ≈ (1 / (N Δt)) Σ (ΔX_i)^2
    #   b   ≈ (1 / (N Δt)) Σ ΔX_i
    b_hat      = dX.sum() / (N * dt)
    sigma2_hat = (dX**2).sum() / (N * dt)
    sigma_hat  = float(np.sqrt(sigma2_hat))

    print("\n[Model-based baseline: parameter estimation]")
    print(f"  N           = {N}")
    print(f"  b_hat       = {b_hat: .6f}   (true b = {mu: .6f})")
    print(f"  sigma_hat   = {sigma_hat: .6f}   (true sigma = {sigma: .6f})\n")

    return b_hat, sigma_hat


def collect_buffer(x_min, x_max, T, dt, batch_size, seed=None):
    """
    Sample one-step transitions (X_t, X_{t+dt}) for the SDE
        dX_t = mu dt + sigma dW_t
    with initial state X_0 ~ Uniform[x_min, x_max].

    Returns
    -------
    buffer : list of (x_t, x_tpdt) pairs (floats)
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    buffer = []
    sqrt_dt = np.sqrt(dt)

    # X ~ rho = Uniform[x_min, x_max]
    X = rng.uniform(low=x_min, high=x_max, size=batch_size)

    for _ in range(n_steps):
        # Brownian increments for each sample in the batch
        dW = rng.normal(loc=0.0, scale=sqrt_dt, size=batch_size)
        X1 = X + mu * dt + sigma * dW

        # store pairs (X_t, X_{t+dt})
        buffer.extend(zip(X, X1))

        # if you want a *path*, uncomment the next line;
        # for i.i.d. one-step samples (as in your original code), keep X fixed
        # X = X1

    return buffer 

T = 20.0
dt = 0.02

batch_size = 256

buffer = collect_buffer(x_min_num, x_max_num, T, dt, batch_size, )
mu_, sigma_ = estimate_drift_and_diffusion_from_buffer(buffer, dt)


def running_cost(x):
    x = np.asarray(x)
    return np.where(x >= 0.0, h * x, -p * x)

def l_cost(xi):
    """
    Impulse cost as a function of xi:
      xi > 0: Kp + kp * xi
      xi < 0: Km + km * (-xi)
      xi = 0: 0  (no impulse)
    """
    xi = np.asarray(xi, dtype=float)
    cost = np.zeros_like(xi, dtype=float)
    pos = xi > 0
    neg = xi < 0
    cost[pos] = Kp + kp * xi[pos]
    cost[neg] = Km + km * (-xi[neg])
    # xi == 0 already set to 0
    return cost


def _extend_psi(psi, x_grid, slope_left=None, slope_right=None):
    """
    Linear interpolation on [x_min, x_max] and linear EXTRAPOLATION outside,
    using given slopes (or boundary finite differences by default).
    """
    psi = np.asarray(psi, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)

    if psi.shape != x_grid.shape:
        raise ValueError("psi and x_grid must have the same shape")

    # Default slopes: local finite differences at boundaries
    if slope_left is None:
        slope_left = (psi[1] - psi[0]) / (x_grid[1] - x_grid[0])
    if slope_right is None:
        slope_right = (psi[-1] - psi[-2]) / (x_grid[-1] - x_grid[-2])

    x_min = x_grid[0]
    x_max = x_grid[-1]

    def psi_ext(z):
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z, dtype=float)

        mask_mid   = (z >= x_min) & (z <= x_max)
        mask_left  =  z <  x_min
        mask_right =  z >  x_max

        if np.any(mask_mid):
            out[mask_mid] = np.interp(z[mask_mid], x_grid, psi)

        if np.any(mask_left):
            out[mask_left] = psi[0] + slope_left  * (z[mask_left] - x_min)

        if np.any(mask_right):
            out[mask_right] = psi[-1] + slope_right * (z[mask_right] - x_max)

        return out

    return psi_ext


def classical_N(psi, x_grid, xi_grid, l_cost_func,
                slope_left=None, slope_right=None):
    """
    Return Npsi[i] = min_j [ psi(x_i + xi_j) + l_cost(xi_j) ],
    using a xi-grid and linear extrapolation of psi outside [x_min, x_max].
    """
    psi     = np.asarray(psi, dtype=float)
    x_grid  = np.asarray(x_grid, dtype=float)
    xi_grid = np.asarray(xi_grid, dtype=float)

    if psi.shape != x_grid.shape:
        raise ValueError("psi and x_grid must have same shape")

    psi_ext = _extend_psi(psi, x_grid, slope_left, slope_right)

    # All x_i + xi_j: shape (Nx, Nxi)
    X  = x_grid[:, None]        # (Nx, 1)
    Xi = xi_grid[None, :]       # (1, Nxi)
    Z  = X + Xi                 # (Nx, Nxi)

    # psi(x_i + xi_j)
    psi_Z = psi_ext(Z)          # same shape as Z

    # Add impulse cost and take min over xi
    l_vals = l_cost_func(xi_grid)      # (Nxi,)
    vals   = psi_Z + l_vals[None, :]   # (Nx, Nxi)

    Npsi = vals.min(axis=1)            # (Nx,)
    return Npsi



def build_fd_coeffs(mu, sigma, r, x_grid):
    """
    Coefficients for (L - r I) with central differences on interior points.
    """
    N = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    a = 0.5 * sigma**2 / dx**2
    b = mu / (2.0 * dx)

    main  = np.zeros(N)
    lower = np.zeros(N-1)
    upper = np.zeros(N-1)

    for i in range(1, N-1):
        lower[i-1] = a - b
        main[i]    = -2.0*a - r
        upper[i]   = a + b

    return lower, main, upper, dx

def solve_uncontrolled_value(mu, sigma, r, x_grid, kp, km):
    """
    Solve (L - r) psi + f = 0 with boundary derivative BCs
        V'(x_min) = -kp,  V'(x_max) = km.
    This is used only as initialization psi^0.
    """
    N = len(x_grid)
    lower, main, upper, dx = build_fd_coeffs(mu, sigma, r, x_grid)
    f_vals = running_cost(x_grid)

    main_full  = main.copy()
    lower_full = lower.copy()
    upper_full = upper.copy()

    # Left boundary: (psi_1 - psi_0)/dx = -kp  -> -psi_0 + psi_1 = -kp*dx
    main_full[0]  = -1.0/dx
    upper_full[0] =  1.0/dx

    # Right boundary: (psi_N-1 - psi_N-2)/dx = km -> -psi_{N-2} + psi_{N-1} = km*dx
    lower_full[-1] = -1.0/dx
    main_full[-1]  =  1.0/dx

    A = diags([lower_full, main_full, upper_full], offsets=[-1,0,1], format="csc")

    rhs = -f_vals.copy()
    rhs[0]  = -kp        # corresponds to -kp*dx / dx
    rhs[-1] =  km

    psi0 = spsolve(A, rhs)
    return psi0

def impulse_operator(psi, x_grid, Kp, kp, Km, km):
    """
    Nonlocal operator N psi for linear impulse costs.
    """
    x = x_grid
    psi = np.asarray(psi)
    N_pts = len(x)

    # up: y >= x
    up_aux = kp * x + psi              # kp * y + psi(y)
    up_min = np.empty(N_pts)
    up_min[-1] = up_aux[-1]
    for i in range(N_pts-2, -1, -1):
        up_min[i] = min(up_aux[i], up_min[i+1])
    up_value = Kp - kp * x + up_min    # Kp - kp*x + min_j>=i (kp x_j + psi_j)

    # down: y <= x
    dn_aux = psi - km * x              # psi(y) - km*y
    dn_min = np.empty(N_pts)
    dn_min[0] = dn_aux[0]
    for i in range(1, N_pts):
        dn_min[i] = min(dn_aux[i], dn_min[i-1])
    dn_value = Km + km * x + dn_min    # Km + km*x + min_j<=i (psi_j - km x_j)

    return np.minimum(up_value, dn_value)

def gs_sweep(psi, Npsi, lower, main, upper, f_vals, dx, kp, km):
    """
    One projected Gauss-Seidel sweep:
        psi_i <- min( psi_i^PDE, Npsi_i ) for interior i,
    with derivative BCs projected at boundaries.
    """
    N = len(psi)
    psi_new = psi.copy()

    # interior points
    for i in range(1, N-1):
        psi_pde = -(f_vals[i] + lower[i-1]*psi_new[i-1] + upper[i]*psi[i+1]) / main[i]
        psi_new[i] = min(psi_pde, Npsi[i])

    # left boundary: V'(x_min) = -kp -> psi_0 = psi_1 + kp*dx
    psi_bc_left = psi_new[1] + kp*dx
    psi_new[0] = min(psi_bc_left, Npsi[0])

    # right boundary: V'(x_max) = km -> psi_{N-1} = psi_{N-2} + km*dx
    psi_bc_right = psi_new[-2] + km*dx
    psi_new[-1] = min(psi_bc_right, Npsi[-1])

    return psi_new

def policy_iteration_classical(mu, sigma, r,
                               x_grid,
                               Kp, kp, Km, km,
                               outer_iters=150,
                               inner_iters=150,
                               use_xi_grid=True,
                               verbose=True):
    """
    Classical policy-iteration baseline for the QVI:
        min{ L psi - r psi + f, N psi - psi } = 0.

    If use_xi_grid=True (default), N is computed via the xi-grid operator
    classical_N with extrapolation outside [x_min, x_max].
    If use_xi_grid=False, falls back to the truncated-domain impulse_operator.
    """
    x_grid = np.asarray(x_grid)
    f_vals = running_cost(x_grid)
    lower, main, upper, dx = build_fd_coeffs(mu, sigma, r, x_grid)

    # initialization: no-impulse value
    psi = solve_uncontrolled_value(mu, sigma, r, x_grid, kp, km)

    # xi-grid for impulses (chosen wider than [x_min, x_max])
    if use_xi_grid:
        # tune these as you like; key is |xi| larger than your x-window
        xi_min, xi_max, Nxi = -6.0, 6.0, 12001
        xi_grid = np.linspace(xi_min, xi_max, Nxi)

    for n in range(1, outer_iters+1):
        psi_old = psi.copy()

        # nonlocal term based on previous iterate
        if use_xi_grid:
            Npsi = classical_N(psi_old, x_grid, xi_grid, l_cost)
        else:
            Npsi = impulse_operator(psi_old, x_grid, Kp, kp, Km, km)

        # a few GS sweeps
        for _ in range(inner_iters):
            psi = gs_sweep(psi, Npsi, lower, main, upper, f_vals, dx, kp, km)

        diff = np.max(np.abs(psi - psi_old))

        if verbose and (n % 20 == 0 or n == 1):
            print(f"[Classical PI] iter {n:03d}, sup-norm diff = {diff:.3e}")

    return psi


# ---- run baseline and compare to analytic solution ----
psi_PI = policy_iteration_classical(mu_, sigma_, r,
                                    x_grid,
                                    Kp, kp, Km, km,
                                    outer_iters=30,
                                    inner_iters=30,
                                    # optional, just to be explicit:
                                    use_xi_grid=True,
                                    verbose=True)

np.save("psi_PI.npy", psi_PI)

V_grid = np.array([V(x) for x in x_grid])
sup_err = np.max(np.abs(psi_PI - V_grid))
print(f"Sup-norm error vs analytic classical V: {sup_err:.3e}")


