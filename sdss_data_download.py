import os
from astropy.io import fits
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import Planck18 as cosmo


# Your Results folder
results_folder = r"F:\st xavier's\S3\Galaxies\Project\Results"

# Create folder if it doesn't exist
os.makedirs(results_folder, exist_ok=True)

# Change Python's working directory
os.chdir(results_folder)

print("Current working directory:", os.getcwd())




query = """
SELECT TOP 100 ra, dec, z, class, specobjid
FROM SpecObj
WHERE class = 'GALAXY' AND z BETWEEN 0.01 AND 0.1
"""

result = SDSS.query_sql(query)
print(result)

print(result[:5])           # View first 5 rows
print(result.colnames)     # See available columns
print(len(result))         # Number of galaxies
print(result[0])           # View first row




# Create output folder
os.makedirs("spectra", exist_ok=True)

for i, row in enumerate(result, 1):  # Start index at 1
    ra, dec = row['ra'], row['dec']
    try:
        spectra = SDSS.get_spectra(coordinates=SkyCoord(ra*u.deg, dec*u.deg), radius=2*u.arcsec)
        if spectra:
            spectra[0].writeto(f"spectra/spectrum_{i}.fits", overwrite=True)
            print(f"[{i}] ✅ Saved spectrum_{i}.fits")
        else:
            print(f"[{i}] ❌ No spectrum at RA={ra}, Dec={dec}")
    except Exception as e:
        print(f"[{i}] ❌ Error at RA={ra}, Dec={dec}: {e}")


# to open one of the image
hdul = fits.open("F:\st xavier's\S3\Galaxies\Project\Results\spectra\spectrum_2.fits")
hdul.info()

#to view the data
data = hdul[1].data
wavelength = 10 ** data['loglam']  # Convert log10(lambda) to Angstroms
flux = data['flux']

plt.figure(figsize=(10,5))
plt.plot(wavelength, flux, color='blue')
plt.xlabel("Wavelength (Å)")
plt.ylabel("Flux (10^-17 erg/s/cm²/Å)")
plt.title("SDSS Optical Spectrum")
plt.show()
plt.savefig(os.path.join(results_folder, "spectrum_plot(2nd one).png"))



c_kms = 299792.458  # km/s

IN_DIR  = r"F:\st xavier's\S3\Galaxies\Project\Results\spectra"
OUT_CSV = r"F:\st xavier's\S3\Galaxies\Project\Results\galaxy_lines_velocities.csv"

def safe_line_flux(wl, fl, ivar, z, line_rest, win=20.0, cont1=(0,0), cont2=(0,0)):
    """
    Continuum-subtracted box integration around observed line center.
    wl [Å], fl [1e-17 erg s^-1 cm^-2 Å^-1], ivar [1/flux^2 per Å].
    Returns (F_line, F_err) in *erg s^-1 cm^-2* (not 1e-17 units).
    """
    m = np.isfinite(wl) & np.isfinite(fl) & np.isfinite(ivar) & (ivar > 0)
    if m.sum() < 5:
        return np.nan, np.nan
    wl, fl, ivar = wl[m], fl[m], ivar[m]

    line_obs = line_rest * (1.0 + z)
    L = (wl > (line_obs - win)) & (wl < (line_obs + win))

    # If sidebands not given, choose simple defaults ±(40–60 Å) from center
    if cont1 == (0,0) and cont2 == (0,0):
        cont1 = (line_rest - 120, line_rest - 80)
        cont2 = (line_rest + 80,  line_rest + 120)

    C1 = (wl > cont1[0]*(1+z)) & (wl < cont1[1]*(1+z))
    C2 = (wl > cont2[0]*(1+z)) & (wl < cont2[1]*(1+z))
    C  = C1 | C2

    if L.sum() < 3 or C.sum() < 3:
        return np.nan, np.nan

    cont_level = np.nanmedian(fl[C])               # flat continuum
    fl_sub = fl[L] - cont_level
    wl_line = wl[L]

    # Integrated flux (in 1e-17 units) then convert to erg s^-1 cm^-2
    F_line_1e17 = np.trapz(fl_sub, wl_line)
    F_line = F_line_1e17 * 1e-17

    # Rough uncertainty: propagate inverse variance through the integration
    # sigma_F^2 ≈ Σ ( (Δλ_i * σ_i)^2 ), with σ_i = 1/sqrt(ivar_i)
    dlam = np.gradient(wl_line)
    sigma_pix = 1.0 / np.sqrt(ivar[L])
    var_F = np.sum((dlam * sigma_pix)**2) * (1e-17**2)
    F_err = np.sqrt(var_F)

    return float(F_line), float(F_err)

def in_coverage(z, line_rest, wl_min=3800.0, wl_max=9200.0):
    lam = line_rest*(1+z)
    return (lam > wl_min) and (lam < wl_max)

def read_one(path):
    with fits.open(path, memmap=False) as hdul:
        coadd = hdul[1].data
        wl   = 10**coadd['loglam']
        flux = coadd['flux']      # 1e-17 erg s^-1 cm^-2 Å^-1
        ivar = coadd['ivar']      # inverse variance of flux

        sp   = hdul[2].data
        z    = float(sp['z'][0]) if 'z' in sp.names else np.nan
        ra   = float(sp['ra'][0]) if 'ra' in sp.names else np.nan
        dec  = float(sp['dec'][0]) if 'dec' in sp.names else np.nan
        cls  = sp['class'][0] if 'class' in sp.names else 'UNKNOWN'
    return wl, flux, ivar, z, ra, dec, cls

rows = []
for fname in sorted(os.listdir(IN_DIR)):
    if not fname.lower().endswith(".fits"):
        continue
    path = os.path.join(IN_DIR, fname)
    try:
        wl, fl, ivar, z, ra, dec, cls = read_one(path)
        if not np.isfinite(z):
            rows.append({'file': fname, 'ra': ra, 'dec': dec, 'class': cls,
                         'z': np.nan, 'v_cz_kms': np.nan, 'v_rel_kms': np.nan,
                         'dL_Mpc': np.nan,
                         'F_Ha': np.nan, 'eF_Ha': np.nan,
                         'F_Hb': np.nan, 'eF_Hb': np.nan,
                         'F_OIII5007': np.nan, 'eF_OIII5007': np.nan,
                         'L_Ha': np.nan, 'L_Hb': np.nan, 'L_OIII5007': np.nan})
            continue

        # velocities
        v_cz   = c_kms * z
        v_rel  = c_kms * ((1+z)**2 - 1)/((1+z)**2 + 1)

        # distance
        dL = cosmo.luminosity_distance(z).to('Mpc').value
        dL_cm = cosmo.luminosity_distance(z).to('cm').value

        # line fluxes (skip if outside coverage)
        F_Ha=F_Hb=F_O3 = np.nan
        eHa=eHb=eO3 = np.nan

        if in_coverage(z, 6562.8):
            F_Ha, eHa = safe_line_flux(wl, fl, ivar, z, 6562.8, win=20, cont1=(6500,6520), cont2=(6605,6625))
        if in_coverage(z, 4861.3):
            F_Hb, eHb = safe_line_flux(wl, fl, ivar, z, 4861.3, win=15, cont1=(4800,4820), cont2=(4890,4910))
        if in_coverage(z, 5006.84):
            F_O3, eO3 = safe_line_flux(wl, fl, ivar, z, 5006.84, win=15, cont1=(4950,4970), cont2=(5035,5055))

        # luminosities
        L_Ha = 4*np.pi*dL_cm**2*F_Ha if np.isfinite(F_Ha) else np.nan
        L_Hb = 4*np.pi*dL_cm**2*F_Hb if np.isfinite(F_Hb) else np.nan
        L_O3 = 4*np.pi*dL_cm**2*F_O3 if np.isfinite(F_O3) else np.nan

        rows.append({
            'file': fname, 'ra': ra, 'dec': dec, 'class': cls,
            'z': z, 'v_cz_kms': v_cz, 'v_rel_kms': v_rel, 'dL_Mpc': dL,
            'F_Ha': F_Ha, 'eF_Ha': eHa,
            'F_Hb': F_Hb, 'eF_Hb': eHb,
            'F_OIII5007': F_O3, 'eF_OIII5007': eO3,
            'L_Ha': L_Ha, 'L_Hb': L_Hb, 'L_OIII5007': L_O3
        })

    except Exception as e:
        rows.append({'file': fname, 'ra': np.nan, 'dec': np.nan, 'class': 'ERROR',
                     'z': np.nan, 'v_cz_kms': np.nan, 'v_rel_kms': np.nan, 'dL_Mpc': np.nan,
                     'F_Ha': np.nan, 'eF_Ha': np.nan,
                     'F_Hb': np.nan, 'eF_Hb': np.nan,
                     'F_OIII5007': np.nan, 'eF_OIII5007': np.nan,
                     'L_Ha': np.nan, 'L_Hb': np.nan, 'L_OIII5007': np.nan,
                     'error': str(e)})

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV)
print(df[['z','v_cz_kms','dL_Mpc','F_Ha','L_Ha']].head())
