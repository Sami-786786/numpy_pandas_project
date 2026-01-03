import numpy as np
import pandas as pd

# -----------------------------
# Dataset
# -----------------------------
data = {
    "ProductID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
    "ProductName": [
        "Milk", "Bread", "Eggs", "Juice", "Chocolate",
        "Cheese", "Apple", "Banana", "Rice", "Pasta",
        "Butter", "Yogurt", "Water", "Tea", "Coffee" ],
    "Category": [
        "Dairy", "Bakery", "Dairy", "Beverages", "Snacks",
        "Dairy", "Fruits", "Fruits", "Grains", "Grains",
        "Dairy", "Dairy", "Beverages", "Beverages", "Beverages"  ],
    "Price": [
        200, 70, 30, np.nan, 50,
        150, 350, 100, 270, 1200,
        600, 90, 10, 80, 230],
    "QuantitySold": [
        120, 60, 200, 50, np.nan,
        30, 100, 120, 40, 35,
        25, 60, 300, 45, 20  ],
    "SaleDate": [
        "2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03",
        "2026-01-03", "2026-01-04", "2026-01-04", "2026-01-05", "2026-01-05",
        "2026-01-06", "2026-01-06", "2026-01-07", "2026-01-07", "2026-01-08"  ],
    "CustomerID": [
        1001, 1002, 1003, 1001, 1004,
        1005, 1006, 1002, 1007, 1008,
        1009, 1010, 1011, 1003, 1012  ],
    "Discount": [
        0.1, 0.0, 0.05, 0.2, 0.0,
        0.15, 0.0, np.nan, 0.1, 0.0,
        0.2, 0.0, 0.05, 0.1, 0.0 ]
}

df = pd.DataFrame(data)

# -----------------------------
# Missing Values Handling
# -----------------------------
print("----- Missing Values Handling -----")
df['Price'] = df['Price'].fillna(df['Price'].mean())
df['QuantitySold'] = df['QuantitySold'].fillna(df['QuantitySold'].mean())
df['Discount'] = df['Discount'].fillna(df['Discount'].mean())
print(df.isnull().sum())

# -----------------------------
# Feature Engineering
# -----------------------------
print("\n----- Feature Engineering -----")
df["TotalSale"] = df['Price'] * df['QuantitySold'] * (1 - df['Discount'])
df["Profit"] = df['TotalSale'] * 0.2
df['HighSale'] = df['QuantitySold'] > 100
print(df[['ProductName','TotalSale','Profit','HighSale']].head(10))

# -----------------------------
# Filtering / Conditional Selection
# -----------------------------
print("\n----- Filtering / Conditional Selection -----")
high_sold = df[df['QuantitySold'] > 100]
print("Products sold > 100 units:\n", high_sold[['ProductName','QuantitySold']])

total_sale_sold = df[df['TotalSale'] > 150]
print("Products with TotalSale > 150:\n", total_sale_sold[['ProductName','TotalSale']])

discount_products = df[df['Discount'] > 0]
print("Products with Discount > 0:\n", discount_products[['ProductName','Discount']])

filtered = df[(df['Price'] > 50) & (df['Category'] == "Beverages")]
print("Beverages with Price > 50:\n", filtered[['ProductName','Price','Category']])

# -----------------------------
# Sorting
# -----------------------------
print("\n----- Sorting -----")
print("Top 5 by TotalSale:\n", df.sort_values(by="TotalSale", ascending=False)[['ProductName','TotalSale']].head())
print("Top 5 by QuantitySold:\n", df.sort_values(by="QuantitySold", ascending=False)[['ProductName','QuantitySold']].head())
print("Top 3 by Profit:\n", df.sort_values(by="Profit", ascending=False)[['ProductName','Profit']].head(3))

# -----------------------------
# Grouping / Aggregation
# -----------------------------
print("\n----- Grouping & Aggregation -----")
grouped = df.groupby("Category")["TotalSale"].sum()
print("TotalSale by Category:\n", grouped)

average = df.groupby("Category")["TotalSale"].mean()
print("Average TotalSale by Category:\n", average)

customer = df.groupby("CustomerID")["TotalSale"].sum()
print("TotalSale by Customer:\n", customer.head())

day = df.groupby("SaleDate")["TotalSale"].sum()
print("TotalSale by Day:\n", day)

# -----------------------------
# NumPy Practise
# -----------------------------
print("\n----- NumPy Practise -----")
Price = np.array(df['Price'])
print("Price median:", np.median(Price))
print("Price mean:", np.mean(Price))
print("Price std:", np.std(Price))

QuantitySold = np.array(df['QuantitySold'])
mask = QuantitySold > 100
print("QuantitySold > 100 mask:", mask)
print("Filtered QuantitySold > 100:", QuantitySold[mask])

TotalSale = np.array(df['TotalSale'])
print("Max TotalSale:", np.max(TotalSale))
print("Min TotalSale:", np.min(TotalSale))
print("Sum TotalSale:", np.sum(TotalSale))
