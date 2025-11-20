import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, lognorm
from sklearn.mixture import GaussianMixture

def plot_distribution(csv_path, fig_path, column="data", dist=None, bins=30, figsize=(10,6)):
    """
    Read CSV data, fit or use specified distribution, and plot histogram + PDF.

    Parameters
    ----------
    csv_path : str
        Path to CSV file.
    column : str
        Column name to plot.
    dist : str or None
        Specify distribution: "normal", "lognormal", "gmm". If None, function fits all and selects best.
    bins : int
        Number of bins for histogram.
    figsize : tuple
        Figure size.
    """
    # --- Read data ---
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in CSV")
    data = df[column].dropna().values

    # --- Fit Normal ---
    mu_norm, std_norm = norm.fit(data)
    loglik_norm = np.sum(norm.logpdf(data, mu_norm, std_norm))
    aic_norm = 2*2 - 2*loglik_norm

    # --- Fit Lognormal ---
    try:
        shape, loc, scale = lognorm.fit(data[data>0], floc=0)
        loglik_logn = np.sum(lognorm.logpdf(data[data>0], shape, loc=loc, scale=scale))
        aic_logn = 2*3 - 2*loglik_logn
    except:
        aic_logn = np.inf

    # --- Fit GMM (2 components) ---
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    gmm.fit(data.reshape(-1,1))
    loglik_gmm = gmm.score(data.reshape(-1,1)) * len(data)
    n_params = 2*2 + 1
    aic_gmm = 2*n_params - 2*loglik_gmm

    # --- Determine best distribution ---
    dist_dict = {'normal': aic_norm, 'lognormal': aic_logn, 'gmm': aic_gmm}
    
    if dist is None:
        best_dist = min(dist_dict, key=dist_dict.get)
    else:
        dist = dist.lower()
        if dist not in dist_dict:
            raise ValueError(f"Invalid dist '{dist}'. Choose from 'normal', 'lognormal', 'gmm'.")
        best_dist = dist

    # --- Generate PDF ---
    x_vals = np.linspace(data.min(), data.max(), 500)
    if best_dist == 'normal':
        pdf_vals = norm.pdf(x_vals, mu_norm, std_norm)
        param_text = f"Normal\nμ={mu_norm:.2f}, σ={std_norm:.2f}"
    elif best_dist == 'lognormal':
        pdf_vals = lognorm.pdf(x_vals, shape, loc=loc, scale=scale)
        param_text = f"Lognormal\nμ={(scale * np.exp(shape**2 / 2)):.2f}, σ={(scale * np.exp(shape**2 / 2)*np.sqrt(np.exp(shape**2) - 1)):.2f}"
    else:  # GMM
        pdf_vals = np.exp(gmm.score_samples(x_vals.reshape(-1,1)))
        means = gmm.means_.flatten()
        weights = gmm.weights_
        stds = np.sqrt(gmm.covariances_).flatten()
        param_text = f"GMM\nμ1={means[0]:.2f}, σ1={stds[0]:.2f}, w1={weights[0]:.2f}\nμ2={means[1]:.2f}, σ2={stds[1]:.2f}, w2={weights[1]:.2f}"

    # --- Plot ---
    plt.figure(figsize=figsize)
    sns.histplot(data, bins=bins, stat='density', color='skyblue', edgecolor='black', alpha=0.6, label='Histogram')
    plt.plot(x_vals, pdf_vals, 'r-', lw=2, label=f"{best_dist.upper()} PDF")
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.title(f"Distribution: {best_dist.upper()}")
    plt.text(0.95, 0.95, param_text, transform=plt.gca().transAxes,
             fontsize=12, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.6))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=1000)

    return best_dist


import os

csv_folder = r"C:\Users\mbavandpour\OneDrive - University of Nevada, Reno\PhD\FilesShouldShareWithBoxFolder\Facundo_Research\Code Repository\data\era5_wind_postprocessing_5by5_dayhours"
figures_folder = os.path.join(csv_folder, "figures")

if not os.path.exists(figures_folder):
    os.mkdir(figures_folder)

csv_files = os.listdir(csv_folder)

for csv in csv_files:
    print(csv)

    if csv.endswith("_vel.csv"):
        csv_path = os.path.join(csv_folder, csv)
        plot_distribution(csv_path, os.path.join(figures_folder, (csv.split(".")[0]+".png")), column="data", bins=40) # "normal", "lognormal", "gmm"
    
    if csv.endswith("_ang.csv"):
        csv_path = os.path.join(csv_folder, csv)
        plot_distribution(csv_path, os.path.join(figures_folder, (csv.split(".")[0]+".png")), column="data", bins=40)