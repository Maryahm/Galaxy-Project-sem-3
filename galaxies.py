import os

# Your Results folder
results_folder = r"F:\st xavier's\S3\Galaxies\Project\Results"

# Create folder if it doesn't exist
os.makedirs(results_folder, exist_ok=True)

# Change Python's working directory
os.chdir(results_folder)

print("Current working directory:", os.getcwd())



from astroquery.sdss import SDSS

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


from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

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




