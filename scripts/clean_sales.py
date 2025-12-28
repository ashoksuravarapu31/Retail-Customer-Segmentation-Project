import pandas as pd

# Load Excel dataset
df = pd.read_excel("retail_sales.xlsx")

# Remove rows with missing CustomerID (very important)
df.dropna(subset=["CustomerID"], inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Fill missing descriptions
df["Description"].fillna("No Description", inplace=True)

# Calculate Total Price
df["Total"] = df["Quantity"] * df["UnitPrice"]

# Convert InvoiceDate column to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

# Save cleaned data to CSV for SQL import
df.to_csv("clean_sales.csv", index=False)

print("Retail data cleaned successfully!")
