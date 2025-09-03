import pandas as pd
import os
import numpy as np
from astropy.cosmology import Planck18 as cosmo

# Set the correct path to your Results folder
results_folder = r"F:\st xavier's\S3\Galaxies\Project\Results"
spectra_dir = os.path.join(results_folder, "spectra")

# Input and output file paths
flux_file = os.path.join(results_folder, "measured_flux.csv")
output_csv = os.path.join(results_folder, "luminosities.csv")

# Load the flux data
flux_df = pd.read_csv(flux_file)

# List to store results
luminosity_data = []

# Calculate luminosity for each galaxy
for _, row in flux_df.iterrows():
    z = row['z']
    
    # Calculate luminosity distance in cm
    DL = cosmo.luminosity_distance(z).to('cm').value
    
    # Convert flux to CGS units (erg/s/cm^2)
    F = row['F_Halpha'] * 1e-17  # SDSS flux units → CGS
    
    # Luminosity calculation: L = 4 * π * D_L^2 * F
    L = 4 * np.pi * (DL ** 2) * F
    
    luminosity_data.append({
        "file": row['file'],
        "z": z,
        "D_L(cm)": DL,
        "L_Halpha(erg/s)": L
    })

# Save results to CSV
pd.DataFrame(luminosity_data).to_csv(output_csv, index=False)
print(f"Saved luminosities to: {output_csv}")
