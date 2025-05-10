import pandas as pd
url= "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(url, nrows=100, sep='\t',encoding="utf-8")


print(df.columns.tolist())
