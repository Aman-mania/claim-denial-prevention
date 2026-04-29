import pandas as pd

# Bronze (raw, as ingested)
df = pd.read_parquet("data/bronze/claims/claims_bronze.parquet")
print(df.head(20))
print(df.info())

# Silver (cleaned, with flags)
df = pd.read_parquet("data/silver/claims/claims_silver.parquet")
print(df[df["proc_no_diag"] == True].head(10))   # filter for violations