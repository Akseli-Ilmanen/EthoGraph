import hashlib
from pathlib import Path

paths = [
    r"C:\Users\aksel\Documents\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250526_01\behav\Trial_data3.nc",
    r"C:\Users\aksel\Documents\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250527_01\behav\Trial_data3.nc",
    r"C:\Users\aksel\Documents\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250527_02\behav\Trial_data3.nc",
    r"C:\Users\aksel\Documents\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250526_02\behav\Trial_data3.nc",
]
for p in paths:
    pp = Path(p)
    digest = hashlib.sha1(str(pp.resolve()).encode("utf-8")).hexdigest()[:8]
    print(digest, p)
