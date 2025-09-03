from astropy.io import fits
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

results_folder = "results"  # Set this to your actual results folder path
spectra_dir = os.path.join(results_folder, "spectra")
output_csv = os.path.join(results_folder, "measured_redshifts.csv")

rest_Halpha = 6562.8  # Angstroms

results = []

for file in sorted(os.listdir(spectra_dir)):
    if not file.endswith(".fits"):
        continue
    
    path = os.path.join(spectra_dir, file)
    with fits.open(path) as hdul:
        data = hdul[1].data
        spec_info = hdul[2].data
        wavelength = 10 ** data['loglam']
        flux = data['flux']
        z_sdss = spec_info['z'][0]
        
        # Find the peak near expected H-alpha
        expected_lambda = rest_Halpha * (1 + z_sdss)
        mask = (wavelength > expected_lambda - 50) & (wavelength < expected_lambda + 50)
        if np.sum(mask) > 0:
            peak_index = np.argmax(flux[mask])
            observed_lambda = wavelength[mask][peak_index]
            z_measured = (observed_lambda - rest_Halpha) / rest_Halpha
        else:
            z_measured = np.nan
        
        results.append({
            "file": file,
            "z_SDSS": z_sdss,
            "z_measured": z_measured,
            "delta_z": z_measured - z_sdss
        })

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Saved measured redshifts to: {output_csv}")




# Path to your Results folder
results_folder = r"F:\st xavier's\S3\Galaxies\Project\Results"
input_csv = os.path.join(results_folder, "measured_redshifts.csv")

# Load the redshift data
df = pd.read_csv(input_csv)
print(f"Loaded {len(df)} galaxies")

# Drop rows with missing measurements
df_clean = df.dropna(subset=['z_measured'])

# # ===========================
# # 1. Plot Measured vs SDSS z
# # ===========================
# plt.figure(figsize=(8, 6))
# plt.scatter(df_clean['z_SDSS'], df_clean['z_measured'], color='blue', alpha=0.7, label="Galaxies")

# # Line y=x for perfect agreement
# z_min = min(df_clean['z_SDSS'].min(), df_clean['z_measured'].min())
# z_max = max(df_clean['z_SDSS'].max(), df_clean['z_measured'].max())
# plt.plot([z_min, z_max], [z_min, z_max], 'r--', label='Perfect Match')

# plt.xlabel("SDSS Redshift (z_SDSS)")
# plt.ylabel("Measured Redshift (z_measured)")
# plt.title("Measured vs SDSS Redshift")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(results_folder, "redshift_comparison.png"), dpi=300)
# plt.show()

# # ===========================
# # 2. Plot Δz Distribution
# # ===========================
# plt.figure(figsize=(8, 5))
# plt.hist(df_clean['delta_z'], bins=20, color='purple', alpha=0.7, edgecolor='black')
# plt.axvline(0, color='red', linestyle='--', label='Perfect Match')

# plt.xlabel("Δz = z_measured - z_SDSS")
# plt.ylabel("Number of Galaxies")
# plt.title("Distribution of Redshift Differences")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(results_folder, "delta_z_distribution.png"), dpi=300)
# plt.show()

# # ===========================
# # 3. Residual Plot
# # ===========================
# plt.figure(figsize=(8, 6))
# plt.scatter(df_clean['z_SDSS'], df_clean['delta_z'], color='green', alpha=0.7)
# plt.axhline(0, color='red', linestyle='--', label='No Difference')

# plt.xlabel("SDSS Redshift (z_SDSS)")
# plt.ylabel("Δz (Measured - SDSS)")
# plt.title("Redshift Residuals")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(results_folder, "redshift_residuals.png"), dpi=300)
# plt.show()

# If no data remains, stop
if df_clean.empty:
    raise ValueError("No valid redshift measurements available for plotting.")

# ==========================
# Define global style
# ==========================
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams["figure.dpi"] = 120

# ==========================
# 1. Measured vs SDSS Redshift
# ==========================
plt.figure(figsize=(8, 6))

# Scatter with color based on delta_z magnitude
scatter = plt.scatter(
    df_clean['z_SDSS'], 
    df_clean['z_measured'], 
    c=np.abs(df_clean['delta_z']), 
    cmap='viridis', 
    s=70, 
    edgecolor='k',
    alpha=0.8,
    label="Galaxies"
)

# Perfect agreement line
z_min = min(df_clean['z_SDSS'].min(), df_clean['z_measured'].min())
z_max = max(df_clean['z_SDSS'].max(), df_clean['z_measured'].max())
plt.plot([z_min, z_max], [z_min, z_max], 'r--', label='Perfect Match')

plt.xlabel("SDSS Redshift (z_SDSS)")
plt.ylabel("Measured Redshift (z_measured)")
plt.title("Measured vs SDSS Redshift", fontsize=14)
plt.legend()
cbar = plt.colorbar(scatter)
cbar.set_label('|Δz| (absolute difference)', rotation=270, labelpad=15)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "redshift_comparison_improved.png"), dpi=300)
plt.show()

# ==========================
# 2. Δz Histogram
# ==========================
plt.figure(figsize=(8, 5))
sns.histplot(df_clean['delta_z'], bins=15, kde=True, color='royalblue')

plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Perfect Match')
plt.xlabel("Δz = z_measured - z_SDSS")
plt.ylabel("Number of Galaxies")
plt.title("Distribution of Redshift Differences (Δz)", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "delta_z_distribution_improved.png"), dpi=300)
plt.show()

# ==========================
# 3. Residual Plot (Δz vs SDSS z)
# ==========================
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df_clean['z_SDSS'],
    y=df_clean['delta_z'],
    hue=np.abs(df_clean['delta_z']),
    palette="coolwarm",
    edgecolor='k',
    s=80
)

plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='No Difference')
plt.xlabel("SDSS Redshift (z_SDSS)")
plt.ylabel("Δz (Measured - SDSS)")
plt.title("Redshift Residuals", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "redshift_residuals_improved.png"), dpi=300)
plt.show()

# ==========================
# 4. Print Summary Stats
# ==========================
mean_delta = df_clean['delta_z'].mean()
std_delta = df_clean['delta_z'].std()
max_delta = df_clean['delta_z'].max()
min_delta = df_clean['delta_z'].min()

print("\nSummary of Δz Statistics:")
print(f"Mean Δz: {mean_delta:.5f}")
print(f"Standard Deviation of Δz: {std_delta:.5f}")
print(f"Max Δz: {max_delta:.5f}")
print(f"Min Δz: {min_delta:.5f}")