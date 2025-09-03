from astropy.io import fits
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
# measure_and_plot_flux.py




results_folder = "results"  # Set this to your actual results folder path
spectra_dir = os.path.join(results_folder, "spectra")
output_csv = os.path.join(results_folder, "measured_flux.csv")

def integrate_flux(wavelength, flux, z, line_rest, window=20):
    line_obs = line_rest * (1 + z)
    mask = (wavelength > line_obs - window) & (wavelength < line_obs + window)
    if np.sum(mask) < 3:
        return np.nan
    return np.trapz(flux[mask], wavelength[mask])

line_data = []
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
        
        F_Halpha = integrate_flux(wavelength, flux, z_sdss, 6562.8)
        F_Hbeta = integrate_flux(wavelength, flux, z_sdss, 4861.3)
        F_OIII = integrate_flux(wavelength, flux, z_sdss, 5006.84)
        
        line_data.append({
            "file": file,
            "z": z_sdss,
            "F_Halpha": F_Halpha,
            "F_Hbeta": F_Hbeta,
            "F_OIII": F_OIII
        })

pd.DataFrame(line_data).to_csv(os.path.join(results_folder, "line_fluxes.csv"), index=False)
print("Saved line fluxes.")



# ==========================================
# 1. Settings
# ==========================================
results_folder = r"F:\st xavier's\S3\Galaxies\Project\Results"  # Update to your path
spectra_dir = os.path.join(results_folder, "spectra")
output_csv = os.path.join(results_folder, "line_fluxes.csv")

# Create Results folder if missing
os.makedirs(results_folder, exist_ok=True)

# ==========================================
# 2. Function to Integrate Flux
# ==========================================
def integrate_flux(wavelength, flux, z, line_rest, window=20):
    """
    Integrate flux around a given spectral line.
    
    Parameters:
        wavelength : array
            Wavelength array in Angstroms.
        flux : array
            Flux array in SDSS units (1e-17 erg/s/cm^2/Å).
        z : float
            Redshift of the galaxy.
        line_rest : float
            Rest-frame wavelength of the line (Angstroms).
        window : float
            Half-width of the integration window (Angstroms).
    
    Returns:
        float: Integrated flux for the line.
    """
    line_obs = line_rest * (1 + z)  # Observed wavelength due to redshift
    mask = (wavelength > line_obs - window) & (wavelength < line_obs + window)
    
    if np.sum(mask) < 3:  # Too few points to integrate
        return np.nan
    
    return np.trapz(flux[mask], wavelength[mask])

# ==========================================
# 3. Loop Through Spectra Files
# ==========================================
line_data = []
total_files = 0

for file in sorted(os.listdir(spectra_dir)):
    if not file.endswith(".fits"):
        continue
    
    total_files += 1
    path = os.path.join(spectra_dir, file)
    
    try:
        with fits.open(path) as hdul:
            data = hdul[1].data          # COADD extension: contains spectrum
            spec_info = hdul[2].data     # SPECOBJ extension: metadata like redshift
            
            wavelength = 10 ** data['loglam']  # Convert log10(λ) → λ in Ångströms
            flux = data['flux']
            z_sdss = spec_info['z'][0]
            
            # Measure fluxes for key lines
            F_Halpha = integrate_flux(wavelength, flux, z_sdss, 6562.8)
            F_Hbeta = integrate_flux(wavelength, flux, z_sdss, 4861.3)
            F_OIII = integrate_flux(wavelength, flux, z_sdss, 5006.84)
            
            line_data.append({
                "file": file,
                "z": z_sdss,
                "F_Halpha": F_Halpha,
                "F_Hbeta": F_Hbeta,
                "F_OIII": F_OIII
            })
    except Exception as e:
        print(f"❌ Error processing {file}: {e}")

# ==========================================
# 4. Save Results to CSV
# ==========================================
df = pd.DataFrame(line_data)
df.to_csv(output_csv, index=False)
print(f"✅ Saved flux measurements for {len(df)} spectra (out of {total_files} files) to {output_csv}")

# ==========================================
# 5. Plotting the Flux Distributions
# ==========================================
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams["figure.dpi"] = 120

# ---------- Histogram for each line ----------
lines = {
    "F_Halpha": "Hα (6562.8 Å)",
    "F_Hbeta": "Hβ (4861.3 Å)",
    "F_OIII": "[OIII] (5006.84 Å)"
}

for col, title in lines.items():
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col].dropna(), bins=20, kde=True, color='royalblue', edgecolor='black')
    plt.xlabel(f"{title} Flux (integrated, 1e-17 erg/s/cm²)")
    plt.ylabel("Number of Galaxies")
    plt.title(f"Distribution of {title} Line Flux")
    plt.tight_layout()
    save_path = os.path.join(results_folder, f"{col}_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved plot: {save_path}")

# ---------- Scatter Plot: Hα vs Redshift ----------
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df['z'],
    y=df['F_Halpha'],
    color='darkorange',
    edgecolor='k',
    s=80,
    alpha=0.8
)
plt.xlabel("Redshift (z)")
plt.ylabel("Hα Flux (integrated, 1e-17 erg/s/cm²)")
plt.title("Hα Flux vs Redshift")
plt.grid(True)
plt.tight_layout()
scatter_path = os.path.join(results_folder, "Halpha_flux_vs_redshift.png")
plt.savefig(scatter_path, dpi=300)
plt.show()
print(f"Saved plot: {scatter_path}")
