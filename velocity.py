# velocities_luminosities.py
import os
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
import matplotlib.pyplot as plt

# ========================
# 1. Set up paths
# ========================
results_folder = r"F:\st xavier's\S3\Galaxies\Project\Results"   # <- Set your actual results folder
flux_csv = os.path.join(results_folder, "line_fluxes.csv")       # Input from your flux script
out_csv = os.path.join(results_folder, "luminosities_with_velocity.csv")  # Final output

# Speed of light in km/s
c_kms = 299792.458

# ========================
# 2. Load flux data
# ========================
df = pd.read_csv(flux_csv)
print(f"Loaded {len(df)} rows from {flux_csv}")

# Check required columns
required_cols = ['z', 'F_Halpha', 'F_Hbeta', 'F_OIII']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# ========================
# 3. Main calculations
# ========================
rows = []
for _, r in df.iterrows():
    file = r.get('file', '')
    z = r['z']

    if not np.isfinite(z):  # Skip if redshift missing
        v_cz = np.nan
        v_rel = np.nan
        dL_Mpc = np.nan
        dL_cm = np.nan
    else:
        # 1) Velocity calculations
        v_cz = c_kms * z  # Approximation: v = cz
        v_rel = c_kms * (((1 + z) ** 2 - 1) / ((1 + z) ** 2 + 1))  # Relativistic velocity

        # 2) Distance
        dL = cosmo.luminosity_distance(z)
        dL_Mpc = dL.value
        dL_cm = dL.to('cm').value

    # Function to compute luminosity for each flux line
    def lum_from_flux(flux_1e17):
        """Convert SDSS flux (10^-17 erg/s/cm^2) to luminosity (erg/s)."""
        if not np.isfinite(flux_1e17) or not np.isfinite(dL_cm):
            return np.nan
        F_cgs = flux_1e17 * 1e-17
        return 4.0 * np.pi * (dL_cm ** 2) * F_cgs

    # 3) Calculate luminosities
    L_Halpha = lum_from_flux(r['F_Halpha'])
    L_Hbeta = lum_from_flux(r['F_Hbeta'])
    L_OIII = lum_from_flux(r['F_OIII'])

    rows.append({
        'file': file,
        'z': z,
        'v_cz_kms': v_cz,
        'v_rel_kms': v_rel,
        'D_L_Mpc': dL_Mpc,
        'D_L_cm': dL_cm,
        'F_Halpha(1e-17)': r['F_Halpha'],
        'F_Hbeta(1e-17)': r['F_Hbeta'],
        'F_OIII(1e-17)': r['F_OIII'],
        'L_Halpha(erg/s)': L_Halpha,
        'L_Hbeta(erg/s)': L_Hbeta,
        'L_OIII(erg/s)': L_OIII
    })

# ========================
# 4. Save to CSV
# ========================
out_df = pd.DataFrame(rows)
out_df.to_csv(out_csv, index=False)

print(f"\nSaved {len(out_df)} rows to: {out_csv}")
print("Columns in output CSV:")
print(out_df.columns.tolist())


# plot_results.py


# ========================
# 1. Load Data
# ========================
results_folder = r"F:\st xavier's\S3\Galaxies\Project\Results"
input_csv = os.path.join(results_folder, "luminosities_with_velocity.csv")

df = pd.read_csv(input_csv)
df_clean = df.dropna(subset=['z', 'v_cz_kms', 'L_Halpha(erg/s)'])

print(f"Loaded {len(df_clean)} valid rows for plotting.")

# ========================
# 2. Velocity vs Redshift
# ========================
plt.figure(figsize=(8, 6))
plt.scatter(df_clean['z'], df_clean['v_cz_kms'], color='blue', alpha=0.7, edgecolor='black')

# Theoretical line v = cz
c_kms = 299792.458
z_range = np.linspace(df_clean['z'].min(), df_clean['z'].max(), 100)
plt.plot(z_range, c_kms * z_range, 'r--', label='v = cz')

plt.xlabel("Redshift (z)", fontsize=12)
plt.ylabel("Velocity (km/s)", fontsize=12)
plt.title("Galaxy Velocity vs Redshift", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "velocity_vs_redshift.png"), dpi=300)
plt.show()

# ========================
# 3. Distance vs Redshift
# ========================
plt.figure(figsize=(8, 6))
plt.scatter(df_clean['z'], df_clean['D_L_Mpc'], color='purple', alpha=0.7, edgecolor='black')

plt.xlabel("Redshift (z)", fontsize=12)
plt.ylabel("Luminosity Distance (Mpc)", fontsize=12)
plt.title("Luminosity Distance vs Redshift", fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "distance_vs_redshift.png"), dpi=300)
plt.show()

# ========================
# 4. Hα Luminosity vs Redshift
# ========================
plt.figure(figsize=(8, 6))
plt.scatter(df_clean['z'], df_clean['L_Halpha(erg/s)'], color='green', alpha=0.7, edgecolor='black')

plt.yscale('log')  # Luminosity often spans many orders of magnitude
plt.xlabel("Redshift (z)", fontsize=12)
plt.ylabel("Hα Luminosity (erg/s)", fontsize=12)
plt.title("Hα Luminosity vs Redshift", fontsize=14, fontweight='bold')
plt.grid(alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "Halpha_luminosity_vs_redshift.png"), dpi=300)
plt.show()

# ========================
# 5. Velocity vs Distance
# ========================
plt.figure(figsize=(8, 6))
plt.scatter(df_clean['D_L_Mpc'], df_clean['v_cz_kms'], color='orange', alpha=0.7, edgecolor='black')

plt.xlabel("Luminosity Distance (Mpc)", fontsize=12)
plt.ylabel("Velocity (km/s)", fontsize=12)
plt.title("Hubble Relation: Velocity vs Distance", fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "velocity_vs_distance.png"), dpi=300)
plt.show()

print(f"✅ All plots saved to: {results_folder}")
