import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("clean_sales.csv")

print("\n===== HEAD =====")
print(df.head())

print("\n===== INFO =====")
print(df.info())

print("\n===== SUMMARY =====")
print(df.describe(include="all"))

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# ---- TOP 10 PRODUCTS REVENUE ----
top_products = df.groupby("Description")["Total"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=top_products.values, y=top_products.index)
plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Product")
plt.tight_layout()
plt.show()

# ---- REVENUE DISTRIBUTION ----
plt.figure(figsize=(8,5))
sns.histplot(df["Total"], bins=30, kde=True)
plt.title("Distribution of Total Revenue per Transaction")
plt.xlabel("Revenue per Transaction")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ---- SALES BY COUNTRY ----
country_sales = df.groupby("Country")["Total"].sum().sort_values(ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x=country_sales.values, y=country_sales.index)
plt.title("Revenue by Country")
plt.xlabel("Total Revenue")
plt.ylabel("Country")
plt.tight_layout()
plt.show()
# ---- REVENUE DISTRIBUTION ----
plt.figure(figsize=(8,5))
sns.histplot(df["Total"], bins=40, kde=True)
plt.title("Distribution of Total Revenue Per Transaction")
plt.xlabel("Revenue Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Month"] = df["InvoiceDate"].dt.to_period("M")
monthly_sales = df.groupby("Month")["Total"].sum().reset_index()
monthly_sales["Month"] = monthly_sales["Month"].astype(str)
plt.figure(figsize=(10,5))
sns.lineplot(data=monthly_sales, x="Month", y="Total", marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
top_customers = df.groupby("CustomerID")["Total"].sum().reset_index()
top_customers = top_customers.sort_values(by="Total", ascending=False).head(10)

print(top_customers)

max_date = df["InvoiceDate"].max()
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (max_date - x.max()).days
}).rename(columns={"InvoiceDate": "Recency"})
rfm["Frequency"] = df.groupby("CustomerID")["InvoiceNo"].nunique()
rfm["Monetary"] = df.groupby("CustomerID")["Total"].sum()
print(rfm.head())
# R Score (lower recency = better)
rfm["R_Score"] = pd.qcut(rfm["Recency"].rank(method="first"), 4, labels=[4,3,2,1])

# F Score (higher frequency = better)
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1,2,3,4])

# M Score (higher monetary = better)
rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 4, labels=[1,2,3,4])



rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)

# ----------------- RFM SEGMENTATION ------------------

def segment_customer(row):
    r, f, m = int(row["R_Score"]), int(row["F_Score"]), int(row["M_Score"])

    if r >= 3 and f >= 3 and m >= 3:
        return "Champion"
    elif r >= 3 and f >= 2:
        return "Loyal Customer"
    elif r >= 2 and f >= 3:
        return "Potential Loyalist"
    elif r == 4 and f == 1:
        return "New Customer"
    elif r <= 2 and f >= 3:
        return "At Risk"
    elif r == 1 and f <= 2:
        return "Lost"
    else:
        return "Needs Attention"

rfm["Segment"] = rfm.apply(segment_customer, axis=1)

print("\n===== CUSTOMER SEGMENTS =====")
print(rfm[["Recency","Frequency","Monetary","RFM_Score","Segment"]].head(20))
plt.figure(figsize=(10,5))
sns.countplot(y=rfm["Segment"], order=rfm["Segment"].value_counts().index)
plt.title("Customer Segmentation Based on RFM")
plt.xlabel("Number of Customers")
plt.ylabel("Segment")
plt.tight_layout()
plt.show()
rfm.to_csv("rfm_output.csv")
