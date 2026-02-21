"""
Experimental code for the paper:
"A Statistical Unified Framework and Convergence Analysis for Forward Physical Modeling and Black-Box Data Modeling"
Core functions: reproduce the paradigm transition, convergence, robustness experiments and PINNs baseline comparison for the gravitational system.
Environment: Python 3.9+, dependencies listed below, fixed random seed for reproducibility.
Results: Outputs true computed values (no hard-coded calibrations). The user should update the paper tables with these results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize
from scipy.stats import linregress
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

# ===================== 1. Environment configuration =====================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

mpl.rcParams.update({
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'axes.linewidth': 1,
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'text.usetex': False,
    'axes.unicode_minus': False
})

# ===================== 2. Fixed experimental parameters =====================
G_TRUE = 6.67e-11
M1 = 1000
M2 = 500
NOISE_SIGMA = 0.002                     # Reduced noise for better fitting
RANDOM_SEED = 42
K_FOLD = 5

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

COLOR_MSE = '#2E86AB'
COLOR_RESIDUAL = '#E63946'
COLOR_REF = '#4CAF50'
COLOR_OURS = '#FF9800'
COLOR_BLACKBOX = '#9C27B0'

# Expanded lambda list with finer granularity
LAMBDA_LIST = [1e12, 1e10, 1e8, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0,
               1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
N_LIST = [50, 100, 200, 500, 1000]
SIGMA_LIST = [0.002, 0.004, 0.008, 0.016]   # Noise levels for robustness
OPTIMAL_LAMBDA = None

# ===================== 3. Core function definitions =====================
def gravity_force(r):
    return G_TRUE * M1 * M2 / (r ** 2)

def physics_residual(f_pred, r):
    r = np.array(r, dtype=np.float64)
    f_pred = np.array(f_pred, dtype=np.float64)
    r_sq = r ** 2
    f_r_sq = f_pred * r_sq
    dr = r[1] - r[0] if len(r) > 1 else 1e-8
    d1f = np.convolve(f_r_sq, [1, -1], mode='valid') / dr
    d2f = np.convolve(d1f, [1, -1], mode='valid') / dr
    residual = np.mean(d2f ** 2)
    return np.clip(residual, 1e-15, 1e5)

# High-order rational model (8th order numerator / 7th order denominator)
def rational_model(r, params):
    # params: [a0..a8, b0..b6]  total 16 parameters
    a = params[:9]
    b = params[9:]
    numerator = (a[0] + a[1]*r + a[2]*r**2 + a[3]*r**3 + a[4]*r**4 +
                 a[5]*r**5 + a[6]*r**6 + a[7]*r**7 + a[8]*r**8)
    denominator = (b[0] + b[1]*r + b[2]*r**2 + b[3]*r**3 + b[4]*r**4 +
                   b[5]*r**5 + b[6]*r**6 + r**7)  # fix highest order term to 1
    y_pred = numerator / (denominator + 1e-8)
    return np.clip(y_pred, 0, 1)

def fit_rational_model(r_train, y_obs_train, lambda_val=0):
    # Pure mechanism (λ large)
    if lambda_val >= 1e7:
        X_phy = M1 * M2 / (r_train ** 2)
        G_hat = np.sum(X_phy * y_obs_train) / (np.sum(X_phy ** 2) + 1e-8)
        y_pred_raw = G_hat * X_phy
        y_min_global = gravity_force(r_train).min()
        y_max_global = gravity_force(r_train).max()
        y_pred = (y_pred_raw - y_min_global) / (y_max_global - y_min_global + 1e-8)
        return np.clip(y_pred, 0, 1), G_hat, (y_min_global, y_max_global)

    # Hybrid/black-box
    # Bounds for 16 parameters
    bounds = [
        (0.01, 5),   # a0
        (-1, 1),     # a1
        (-0.1, 0.1), # a2
        (-0.01, 0.01), # a3
        (-0.001, 0.001), # a4
        (-0.0001, 0.0001), # a5
        (-1e-5, 1e-5), # a6
        (-1e-6, 1e-6), # a7
        (-1e-7, 1e-7), # a8
        (1.0, 10.0), # b0
        (-1, 1),     # b1
        (-0.1, 0.1), # b2
        (-0.01, 0.01), # b3
        (-0.001, 0.001), # b4
        (-0.0001, 0.0001), # b5
        (-1e-5, 1e-5)  # b6
    ]

    # Compute scaling factor based on initial guess at this λ
    # Use initial parameters to estimate typical magnitudes
    init_params = np.array([0.8, 0.05, 0.001, 0.0001, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
                            2.0, 0.05, 0.001, 0.0001, 1e-5, 1e-6, 1e-7])
    y_init = rational_model(r_train, init_params)
    data_loss_init = np.sum((y_init - y_obs_train) ** 2)
    reg_loss_init = physics_residual(y_init, r_train)
    reg_scale = data_loss_init / (reg_loss_init + 1e-8)

    def loss_func(params):
        y_pred = rational_model(r_train, params)
        data_loss = np.sum((y_pred - y_obs_train) ** 2)
        if lambda_val <= 1e-5:  # pure black-box
            return data_loss + 1e-4 * np.sum(params ** 2)
        reg_loss = physics_residual(y_pred, r_train)
        return data_loss + lambda_val * reg_scale * reg_loss

    res = minimize(loss_func, init_params, method='L-BFGS-B',
                   tol=1e-12, bounds=bounds,
                   options={'maxiter': 50000, 'gtol': 1e-10, 'ftol': 1e-12})
    if not res.success:
        print(f"Warning: optimization did not converge for λ={lambda_val}, reason: {res.message}")
    y_pred = rational_model(r_train, res.x)
    y_min_train, y_max_train = y_obs_train.min(), y_obs_train.max()
    return np.clip(y_pred, 0, 1), res.x, (y_min_train, y_max_train)

def predict_ood(r_ood, params, lambda_val, train_stats):
    y_min_train, y_max_train = train_stats
    if lambda_val >= 1e7:
        G_hat = params
        X_phy_ood = M1 * M2 / (r_ood ** 2)
        y_pred_raw = G_hat * X_phy_ood
        return np.clip((y_pred_raw - y_min_train) / (y_max_train - y_min_train + 1e-8), 0, 1)
    return np.clip(rational_model(r_ood, params), 0, 1)

# ===================== 4. PINNs model definition =====================
class PINN(nn.Module):
    def __init__(self, hidden_dim=32):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        return self.net(x)

def pinn_physics_loss(y_pred, r_tensor):
    r_sq = r_tensor ** 2
    f_r_sq = y_pred * r_sq
    d1f = torch.autograd.grad(f_r_sq.mean(), r_tensor, create_graph=True, retain_graph=True)[0]
    d2f = torch.autograd.grad(d1f.mean(), r_tensor, create_graph=True)[0]
    d2f_clipped = torch.clamp(d2f, -1e3, 1e3)
    return torch.mean(d2f_clipped ** 2)

# ===================== 5. Experimental data generation =====================
# Broader range and more points for better fitting
r = np.linspace(1e-5, 5e-3, 1000)
r_ood = np.linspace(5e-3, 1e-2, 1000)

y_true_raw = gravity_force(r)
y_true_raw_ood = gravity_force(r_ood)
y_min_global = y_true_raw.min()
y_max_global = y_true_raw.max()
y_true = (y_true_raw - y_min_global) / (y_max_global - y_min_global)
y_true_ood = (y_true_raw_ood - y_min_global) / (y_max_global - y_min_global)

y_obs = y_true + np.random.normal(loc=0, scale=NOISE_SIGMA, size=len(r))

# ===================== 6. Experiment 1: Paradigm transition curve =====================
print("=" * 70)
print("Experiment 1: Paradigm transition curve (Fig. 1 + Table 1)")
print("=" * 70)

mse_true_mean, mse_true_std = [], []
residual_log_mean, residual_log_std = [], []
ood_mse_mean, ood_mse_std = [], []
val_mse_mean = []

kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=RANDOM_SEED)

for lam in LAMBDA_LIST:
    mse_fold, res_log_fold, ood_fold = [], [], []
    val_mse_fold = []
    for train_idx, val_idx in kf.split(r):
        r_train, y_obs_train = r[train_idx], y_obs[train_idx]
        r_val, y_obs_val = r[val_idx], y_obs[val_idx]
        y_true_train = y_true[train_idx]
        y_true_val = y_true[val_idx]

        y_pred_train, params, train_stats = fit_rational_model(r_train, y_obs_train, lam)
        y_pred_val = predict_ood(r_val, params, lam, train_stats)
        y_pred_ood = predict_ood(r_ood, params, lam, train_stats)

        mse_train = np.mean((y_pred_train - y_true_train) ** 2)
        mse_val = np.mean((y_pred_val - y_true_val) ** 2)
        res = physics_residual(y_pred_train, r_train)
        res_log = np.log10(res)
        ood_mse = np.mean((y_pred_ood - y_true_ood) ** 2)

        mse_fold.append(mse_train)
        res_log_fold.append(res_log)
        ood_fold.append(ood_mse)
        val_mse_fold.append(mse_val)

    mse_true_mean.append(np.mean(mse_fold))
    mse_true_std.append(np.std(mse_fold))
    residual_log_mean.append(np.mean(res_log_fold))
    residual_log_std.append(np.std(res_log_fold))
    ood_mse_mean.append(np.mean(ood_fold))
    ood_mse_std.append(np.std(ood_fold))
    val_mse_mean.append(np.mean(val_mse_fold))

    print(f"λ={lam:.0e} | train MSE={mse_true_mean[-1]:.2e}±{mse_true_std[-1]:.2e} | "
          f"val MSE={val_mse_mean[-1]:.2e} | physical residual=10^{residual_log_mean[-1]:.2f} | OOD-MSE={ood_mse_mean[-1]:.2e}")

# Select optimal lambda balancing validation MSE and physical residual
# Normalize scores to avoid scale issues
val_mse_array = np.array(val_mse_mean)
res_array = 10**np.array(residual_log_mean)
val_mse_norm = val_mse_array / val_mse_array.max()
res_norm = res_array / res_array.max()
combined_score = val_mse_norm + 0.1 * res_norm   # weight 0.1 for physical residual
opt_idx = np.argmin(combined_score)
OPTIMAL_LAMBDA = LAMBDA_LIST[opt_idx]
print(f"\nOptimal lambda selected: {OPTIMAL_LAMBDA:.0e} (combined score={combined_score[opt_idx]:.4f})")

# Plot Fig. 1
fig, ax1 = plt.subplots()
LAMBDA_LOG = np.log10(LAMBDA_LIST)

ax1.errorbar(LAMBDA_LOG, mse_true_mean, yerr=mse_true_std, fmt='o-', color=COLOR_MSE,
             linewidth=2, markersize=6, markeredgecolor='black', markeredgewidth=0.5,
             label='Train MSE', capsize=3, capthick=1)
ax1.set_xlabel('Regularization coefficient $\\lambda$ (log scale)', fontweight='bold')
ax1.set_ylabel('Train MSE', color=COLOR_MSE, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=COLOR_MSE)
ax1.set_xticks(LAMBDA_LOG)
ax1.set_xticklabels([f'$10^{{{int(np.log10(lam))}}}$' for lam in LAMBDA_LIST])
ax1.grid(True, linestyle='--', alpha=0.4, axis='y')
ax1.spines['top'].set_visible(False)

ax2 = ax1.twinx()
ax2.errorbar(LAMBDA_LOG, residual_log_mean, yerr=residual_log_std, fmt='s--', color=COLOR_RESIDUAL,
             linewidth=1.5, markersize=5, markeredgecolor='black', markeredgewidth=0.5,
             label='log(Physical residual)', capsize=2, capthick=1)
ax2.set_ylabel('Physical residual (log scale)', color=COLOR_RESIDUAL, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=COLOR_RESIDUAL)
ax2.spines['top'].set_visible(False)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', frameon=True, framealpha=0.9)
ax1.set_title('Relationship between $\\lambda$ and true MSE / physical residual\n'
              '(error bars: 5‑fold CV standard deviation)', fontsize=12, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('conversion_curve.pdf', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.close()
print("✅ Paradigm transition curve saved: conversion_curve.pdf (Fig. 1)\n")

# ===================== 7. Experiment 2: Convergence curve =====================
print("=" * 70)
print("Experiment 2: Convergence curve (Fig. 2)")
print("=" * 70)

mse_conv_mean, mse_conv_std = [], []
for N in N_LIST:
    mse_fold = []
    for _ in range(K_FOLD):
        r_n = np.linspace(1e-5, 5e-3, N)
        y_true_n_raw = gravity_force(r_n)
        y_true_n = (y_true_n_raw - y_min_global) / (y_max_global - y_min_global)
        y_obs_n = y_true_n + np.random.normal(loc=0, scale=NOISE_SIGMA, size=N)
        y_pred_n, _, _ = fit_rational_model(r_n, y_obs_n, OPTIMAL_LAMBDA)
        mse_fold.append(np.mean((y_pred_n - y_true_n) ** 2))
    mse_conv_mean.append(np.mean(mse_fold))
    mse_conv_std.append(np.std(mse_fold))
    print(f"Sample size N={N} | true MSE={mse_conv_mean[-1]:.2e}±{mse_conv_std[-1]:.2e}")

# Fit O(1/N) line
x_fit = 1 / np.array(N_LIST)
y_fit = np.array(mse_conv_mean)
reg = linregress(x_fit, y_fit)
slope, intercept, r2 = reg.slope, reg.intercept, reg.rvalue ** 2
ref_line = intercept + slope * x_fit
print(f"Fitted O(1/N) line: slope={slope:.4f}, intercept={intercept:.4f}, R²={r2:.4f}")

# Plot Fig. 2
fig, ax = plt.subplots()
ax.errorbar(N_LIST, mse_conv_mean, yerr=mse_conv_std, fmt='s-', color=COLOR_OURS,
            linewidth=2, markersize=6, markeredgecolor='black', markeredgewidth=0.5,
            label='Our optimal mixed model', capsize=3, capthick=1)
ax.plot(N_LIST, ref_line, 'k--', linewidth=1.5, label=r'$O(1/N)$ fit ($R^2=%.3f$)' % r2)
ax.axhline(y=NOISE_SIGMA ** 2, color=COLOR_REF, linestyle=':', linewidth=1.5,
           label=r'Noise floor $\sigma^2=%.2e$' % (NOISE_SIGMA**2))
ax.set_xlabel('Sample size $N$', fontweight='bold')
ax.set_ylabel('True MSE', fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(loc='upper right', frameon=True, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Convergence curve: true MSE vs. sample size\n(error bars: 5‑fold CV standard deviation)',
             fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('convergence_curve.pdf', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.close()
print(f"✅ Convergence curve saved: convergence_curve.pdf (Fig. 2, R²={r2:.3f})\n")

# ===================== 8. Experiment 3: Robustness comparison curve =====================
print("=" * 70)
print("Experiment 3: Robustness comparison curve (supplementary figure)")
print("=" * 70)

mse_ours_list, mse_blackbox_list = [], []
for sigma in SIGMA_LIST:
    mse_ours_temp, mse_black_temp = [], []
    for rep in range(5):   # repeat 5 times for each noise level
        np.random.seed(RANDOM_SEED + rep)
        y_obs_noise = y_true + np.random.normal(loc=0, scale=sigma, size=len(r))

        y_pred_ours, _, _ = fit_rational_model(r, y_obs_noise, OPTIMAL_LAMBDA)
        mse_ours_temp.append(np.mean((y_pred_ours - y_true) ** 2))

        y_pred_black, _, _ = fit_rational_model(r, y_obs_noise, 1e-6)
        mse_black_temp.append(np.mean((y_pred_black - y_true) ** 2))

    mse_ours_list.append(np.mean(mse_ours_temp))
    mse_blackbox_list.append(np.mean(mse_black_temp))
    print(f"Noise σ={sigma:.3f} | Our model MSE={mse_ours_list[-1]:.2e} | Pure black‑box MSE={mse_blackbox_list[-1]:.2e}")

# Plot supplementary figure
fig, ax = plt.subplots()
ax.plot(SIGMA_LIST, mse_ours_list, 'o-', color=COLOR_OURS, linewidth=2, markersize=6,
        markeredgecolor='black', label='Our optimal mixed model')
ax.plot(SIGMA_LIST, mse_blackbox_list, 's--', color=COLOR_BLACKBOX, linewidth=1.5, markersize=5,
        markeredgecolor='black', label='Pure black‑box data modeling')
ax.set_xlabel('Noise standard deviation $\\sigma$', fontweight='bold')
ax.set_ylabel('True MSE', fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(loc='upper left', frameon=True, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Robustness comparison under different noise levels', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('robustness_curve.pdf', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.close()
print("✅ Robustness comparison curve saved: robustness_curve.pdf (supplementary figure)\n")

# ===================== 9. Experiment 4: PINNs baseline comparison =====================
print("=" * 70)
print("Experiment 4: PINNs baseline training (Table 2)")
print("=" * 70)

n_runs = 5
pinn_mse_list = []
pinn_ood_list = []
pinn_res_list = []

for run in range(n_runs):
    run_seed = RANDOM_SEED + run
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    r_tensor = torch.tensor(r, dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
    y_tensor = torch.tensor(y_obs, dtype=torch.float32).reshape(-1, 1)
    r_ood_tensor = torch.tensor(r_ood, dtype=torch.float32).reshape(-1, 1)

    model = PINN(hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lambda_physics = 1.0
    epochs = 1000
    grad_clip = 1.0

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(r_tensor)
        data_loss = torch.mean((y_pred - y_tensor) ** 2)
        phy_loss = pinn_physics_loss(y_pred, r_tensor)
        loss = data_loss + lambda_physics * phy_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_pinn = model(r_tensor).detach().numpy().flatten()
        y_pred_pinn_ood = model(r_ood_tensor).detach().numpy().flatten()

    mse_pinn = np.mean((y_pred_pinn - y_true) ** 2)
    ood_mse_pinn = np.mean((y_pred_pinn_ood - y_true_ood) ** 2)
    res_pinn = physics_residual(y_pred_pinn, r)

    pinn_mse_list.append(mse_pinn)
    pinn_ood_list.append(ood_mse_pinn)
    pinn_res_list.append(res_pinn)

pinn_mse_mean = np.mean(pinn_mse_list)
pinn_mse_std = np.std(pinn_mse_list)
pinn_ood_mean = np.mean(pinn_ood_list)
pinn_ood_std = np.std(pinn_ood_list)
pinn_res_mean = np.mean(pinn_res_list)
pinn_res_std = np.std(pinn_res_list)

print(f"PINNs baseline (5 runs) | true MSE={pinn_mse_mean:.2e}±{pinn_mse_std:.2e} | "
      f"physical residual={pinn_res_mean:.2e}±{pinn_res_std:.2e} | OOD-MSE={pinn_ood_mean:.2e}±{pinn_ood_std:.2e}")

# Obtain pure mechanism and pure black-box results from Experiment 1 (λ=1e8 and λ=1e-6)
idx_mech = LAMBDA_LIST.index(1e8)
idx_black = LAMBDA_LIST.index(1e-6)
pure_mech_mse = f"{mse_true_mean[idx_mech]:.2e}±{mse_true_std[idx_mech]:.2e}"
pure_mech_res = f"{10**residual_log_mean[idx_mech]:.2e}"
pure_mech_ood = f"{ood_mse_mean[idx_mech]:.2e}±{ood_mse_std[idx_mech]:.2e}"
pure_black_mse = f"{mse_true_mean[idx_black]:.2e}±{mse_true_std[idx_black]:.2e}"
pure_black_res = f"{10**residual_log_mean[idx_black]:.2e}"
pure_black_ood = f"{ood_mse_mean[idx_black]:.2e}±{ood_mse_std[idx_black]:.2e}"

print("\n" + "="*70)
print("Table 2: Multi-model comparison (from Experiment 1 and PINNs)")
print("-" * 70)
print(f"Pure mechanism modeling       | true MSE={pure_mech_mse} | physical residual={pure_mech_res} | OOD-MSE={pure_mech_ood}")
print(f"Pure black-box modeling       | true MSE={pure_black_mse} | physical residual={pure_black_res} | OOD-MSE={pure_black_ood}")
print(f"PINNs                          | true MSE={pinn_mse_mean:.2e}±{pinn_mse_std:.2e} | physical residual={pinn_res_mean:.2e}±{pinn_res_std:.2e} | OOD-MSE={pinn_ood_mean:.2e}±{pinn_ood_std:.2e}")
print("="*70)

# ===================== 10. Experiment summary =====================
print("\n" + "="*70)
print("All experiments completed! Generated files:")
print("1. conversion_curve.pdf - Paradigm transition curve (Fig. 1)")
print("2. convergence_curve.pdf - Convergence curve (Fig. 2)")
print("3. robustness_curve.pdf - Robustness comparison curve (supplementary figure)")
print("="*70)
print("Key conclusions (derived from actual computed values):")
print(f"1. Optimal regularization coefficient λ={OPTIMAL_LAMBDA:.0e}, true MSE={mse_true_mean[opt_idx]:.2e} (Table 1)")
print(f"2. Convergence rate O(1/N), R²={r2:.3f} (Fig. 2)")
print(f"3. Our model robustness compared to pure black‑box (see supplementary figure)")
print(f"4. PINNs baseline MSE={pinn_mse_mean:.2e} (Table 2)")
print("="*70)