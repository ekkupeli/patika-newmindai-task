import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

####Mission 1: Data Cleaning and Processing
###Data and Missing Value Control
sales_df= pd.read_csv("datasets\\base\\satis_verisi_5000.csv")
print("""
      -------------------------------------
        Sale Data Example:
      -------------------------------------
      """,sales_df.head())

print("""
      -------------------------------------
        Sale Info:
      -------------------------------------
      """)
print(sales_df.info())

print("""
      -------------------------------------
        Sale Data Value Info:
      -------------------------------------
      """)
print(sales_df["ürün_adi"].value_counts())
print(sales_df["kategori"].value_counts())


customer_df= pd.read_csv("datasets\\base\\musteri_verisi_5000_utf8.csv")
print("""
      -------------------------------------
        Customer Data Example:
      -------------------------------------
      """,customer_df.head())

print("""
      -------------------------------------
        Customer Info:
      -------------------------------------
      """)
print(customer_df.info())

print("""
      -------------------------------------
        Customer Data Value Info:
      -------------------------------------
      """)
print(customer_df["cinsiyet"].value_counts())
print(customer_df["yas"].value_counts())
print(customer_df["sehir"].value_counts())

###Define Outliers

    ##Control of calculations such as mean and standard deviation
#print("""
#      -------------------------------------
#        Describe to Sale Outliers:
#      -------------------------------------
#      """,sales_df.describe())
#
#print("""
#      -------------------------------------
#        Describe to Customer Outliers:
#      -------------------------------------
#      """,customer_df.describe())

    ##1.1-1.2: Function (IQR method) 
    #IQR (Interquartile Range) method is used to detect outliers by measuring the spread of the middle 50% of the data.

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    low_limit = Q1 - 1.5 * IQR
    upp_limit = Q3 + 1.5 * IQR

    ## Filter
    outliers = df[(df[column] < low_limit) | (df[column] > upp_limit)]
    return outliers, low_limit, upp_limit

    ##Identify with relevant colums
customer_yas_outliers, yas_low, yas_upp = detect_outliers_iqr(customer_df, "yas")
customer_harcama_outliers, harcama_low, harcama_upp = detect_outliers_iqr(customer_df, "harcama_miktari")
sales_fiyat_outliers, fiyat_low, fiyat_upp = detect_outliers_iqr(sales_df, "fiyat")
sales_toplam_outliers, toplam_low, toplam_upp = detect_outliers_iqr(sales_df, "toplam_satis")
sales_adet_outliers, adet_low, adet_upp = detect_outliers_iqr(sales_df, "adet")

    ##Summary of each column
outliers_summary = {
    "customer_yas_outliers_count": len(customer_yas_outliers),
    "yas_bounds": (yas_low, yas_upp),
    
    "customer_harcama_outliers_count": len(customer_harcama_outliers),
    "harcama_bounds": (harcama_low, harcama_upp),
    
    "sales_fiyat_outliers_count": len(sales_fiyat_outliers),
    "fiyat_bounds": (fiyat_low, fiyat_upp),
    
    "sales_toplam_outliers_count": len(sales_toplam_outliers),
    "toplam_bounds": (toplam_low, toplam_upp),
    
    "sales_adet_outliers_count": len(sales_adet_outliers),
    "adet_bounds": (adet_low, adet_upp)
}

#print("Summary to outliers:\n",outliers_summary)

"""(print)
  Summary to outliers:
  {'customer_yas_outliers_count': 0, 'yas_bounds': (-8.0, 96.0), 
  'customer_harcama_outliers_count': 0, 'harcama_bounds': (-2407.415, 7514.085), 
  'sales_fiyat_outliers_count': 0, 'fiyat_bounds': (-763.4600000000003, 2242.1800000000003), 
  'sales_toplam_outliers_count': 35, 'toplam_bounds': (-12038.167499999998, 25688.4525),           #outliers are here (up to upp_limit)
  'sales_adet_outliers_count': 0, 'adet_bounds': (-10.0, 30.0)}
"""
    

    ##Graphs (toplam_satis-outliers)
#plt.figure(figsize=(12, 6))
#plt.boxplot(data=sales_df, x=sales_df["toplam_satis"], vert=False, label="total_sales")
#plt.title("Total Sales")
#plt.xlabel("total_sales")
#plt.legend()
#plt.grid(True)
#plt.show()

    ##Equalizing the upp_limit to use outliers without corrupting the dataset 
sales_df["toplam_satis"] = sales_df["toplam_satis"].apply(
    lambda x: min(max(x, toplam_low), toplam_upp))

    #Control of change 
#print(sales_toplam_outliers)
#print(sales_df["toplam_satis"][571])


###1.3: Merge Data with "musteri_id"
merged_df = pd.merge(sales_df, customer_df, on="musteri_id", how="inner")
#merged_df.to_csv("datasets\\custom\\merged_data.csv", index=False)

print("""
      -------------------------------------
        Merged Info:
      -------------------------------------
      """)
print(merged_df.head())


####Mission 2: Time Series Analysis

### Unique control
    #Unique products
unique_products = merged_df["ürün_adi"].unique()
#print(unique_products)
"""(print)
  ['Mouse' 'Kalem' 'Bilgisayar' 'Klima' 'Fırın' 'Defter' 'Çanta' 'Su Şişesi' 'Kulaklık' 'Telefon']
"""
    

    #Unique categories
unique_categories = merged_df["kategori"].unique()
#print(unique_categories)
"""(print)
['Ev Aletleri' 'Kırtasiye' 'Giyim' 'Elektronik' 'Mutfak Ürünleri' 'Kozmetik']
"""


    # Convert the date column to datetime format and check dates
merged_df["tarih"] = pd.to_datetime(merged_df["tarih"])
first_date = merged_df["tarih"].min()
last_date = merged_df["tarih"].max()
#print(f"First date: {first_date}")
#print(f"Last date: {last_date}")
"""(print)
First date: 2022-11-06 00:00:00
Last date: 2024-11-05 00:00:00
"""


###2.1.1: Analysis of Weekly Sales
weekly_sales = merged_df.resample("W-Mon", on="tarih")["toplam_satis"].sum()
weekly_item_sales = merged_df.groupby(["ürün_adi"]).resample("W-Mon", on="tarih")["adet"].sum().unstack(level=0)

    #First 5 weeks, item sales
#print(weekly_item_sales.head())    

    # Graphs (weekly_sales-toplam_satis)
#plt.figure(figsize=(12, 6))
#plt.plot(weekly_sales.index, weekly_sales.values, marker="o",label="weekly_sales")
#plt.title("Weekly Sales")
#plt.xlabel("Date")
#plt.ylabel("Sales")
#plt.xticks(rotation=45)
#plt.legend()
#plt.grid(True)
#plt.show()

    # Graphs (weekly_sales)
#plt.figure(figsize=(12, 6))
#plt.subplot(2,1,1)
#plt.plot(weekly_sales.index, weekly_sales.values, marker="o",label="weekly_sales")
#plt.title("Weekly Sales")
#plt.xlabel("Date")
#plt.ylabel("Sales")
#plt.xticks(rotation=45)
#plt.legend()
#plt.grid(True)
#
#plt.subplot(2,1,2)
#for urun in weekly_item_sales.columns:
#    plt.plot(weekly_item_sales.index, weekly_item_sales[urun], marker="o", label=urun)
#plt.title("Weekly Item Sales")
#plt.xlabel("Date")
#plt.ylabel("Product Sales")
#plt.xticks(rotation=45)
#plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
#plt.grid(True)
#plt.show()


###2.1.2: Analysis of Monthly Sales
monthly_sales = merged_df.resample("ME", on="tarih")["toplam_satis"].sum()
monthly_item_sales = merged_df.groupby(["ürün_adi"]).resample("ME", on="tarih")["adet"].sum().unstack(level=0)

    # Graphs (monthly_sales-toplam_satis)
#plt.figure(figsize=(12, 6))
#plt.plot(monthly_sales.index, monthly_sales.values, marker="o",label="monthly_sales")
#plt.title("Monthly Sales")
#plt.xlabel("Date")
#plt.ylabel("Sales")
#plt.xticks(rotation=45)
#plt.legend()
#plt.grid(True)
#plt.show()

    #Graphs (monthly_sales)
#plt.figure(figsize=(12, 6))
#plt.subplot(2,1,1)
#plt.plot(monthly_sales.index, monthly_sales.values, marker="o",label="monthly_sales")
#plt.title("Monthly Sales")
#plt.xlabel("Date")
#plt.ylabel("Sales")
#plt.xticks(rotation=45)
#plt.legend()
#plt.grid(True)
#
#plt.subplot(2,1,2)
#for urun in monthly_item_sales.columns:
#    plt.plot(monthly_item_sales.index, monthly_item_sales[urun], marker="o", label=urun)
#plt.title("Monthly Item Sales")
#plt.xlabel("Date")
#plt.ylabel("Product Sales")
#plt.xticks(rotation=45)
#plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
#plt.grid(True)
#plt.show()


###2.2.1: Monthly First-Last Days
monthly_first = merged_df.groupby(merged_df["tarih"].dt.to_period("M"))["tarih"].min()
monthly_last = merged_df.groupby(merged_df["tarih"].dt.to_period("M"))["tarih"].max()

print("""
      -------------------------------------
        Monthly First Days:
      -------------------------------------
      """)
print(monthly_first)
print("""
      -------------------------------------
        Monthly Last Days:
      -------------------------------------
      """)
print(monthly_last)


###2.2.2:Weekly Total Item Sales
weekly_item_sales_sum = merged_df.resample("W-Mon", on="tarih")["adet"].sum()

print("""
      -------------------------------------
        Weekly Total Item Sales:
      -------------------------------------
      """)
print(weekly_item_sales_sum.head())

    # Graphs (weekly_item_sales_sum)
#plt.figure(figsize=(12, 6))
#plt.plot(weekly_item_sales_sum.index, weekly_item_sales_sum.values, marker="o",label="weekly_item_sales_sum")
#plt.title("Weekly Total Item Sales")
#plt.xlabel("Date")
#plt.ylabel("Sales")
#plt.xticks(rotation=45)
#plt.legend()
#plt.grid(True)
#plt.show()
