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

#def convert_datetime_to_numeric(date_str):
#    try:
#        # Tarih ve saat bilgisini parçala
#        year, month, _ = date_str.split("-")[:3]  # İlk 2 kısmı al
#        return float(f"{int(year) % 100}.{int(month):02}")  # Yılın son 2 hanesi ve ay
#    except Exception:
#        return None  # Geçersiz format varsa None döndür
#
#sales_df["fiyat"]= sales_df["fiyat"].apply(convert_datetime_to_numeric)

def detect_is_numeric(df, column):
    if not pd.api.types.is_numeric_dtype(df[column]):
        #raise ValueError(f"{column} column must be digital type!")
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df

    # Checks
detect_is_numeric(sales_df, "fiyat")
detect_is_numeric(sales_df, "toplam_satis")
print("""
#      -------------------------------------
#        Check and Fill the NaN:
#      -------------------------------------
#      """)
#print(sales_df["toplam_satis"].info())
#print(sales_df[sales_df["toplam_satis"].isnull()])
sales_df["toplam_satis"]= sales_df["toplam_satis"].fillna(sales_df["fiyat"] * sales_df["adet"])
#print(sales_df["toplam_satis"].info())

#print(sales_df["fiyat"].info())
#print(sales_df[sales_df["fiyat"].isnull()])
sales_df["fiyat"]= sales_df["fiyat"].fillna(sales_df["toplam_satis"] / sales_df["adet"])
#print(sales_df["fiyat"].info())

print("""
#      -------------------------------------
#        Equalizing the Big Numbers:
#      -------------------------------------
#      """)
sales_df["toplam_satis"] = sales_df.apply(
    lambda row: row["fiyat"] * row["adet"] if row["toplam_satis"] > (row["fiyat"] * row["adet"]) else row["toplam_satis"], axis=1)

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
#print("Summary to outliers:\n",sales_fiyat_outliers)
#print("Summary to outliers:\n",sales_toplam_outliers)

"""(print)
  Summary to outliers:  (new dataset)
  {'customer_yas_outliers_count': 0, 'yas_bounds': (-8.0, 96.0), 
  'customer_harcama_outliers_count': 0, 'harcama_bounds': (-2407.415, 7514.085), 
  'sales_fiyat_outliers_count': 0, 'fiyat_bounds': (-744.02125, 2234.5687500000004), 
  'sales_toplam_outliers_count': 467, 'toplam_bounds': (-15448.205000000002, 31851.595),        #outliers are here (up to upp_limit)
  'sales_adet_outliers_count': 0, 'adet_bounds': (-10.0, 30.0)}
  
  Summary to outliers:  (after fill nan)
 {'customer_yas_outliers_count': 0, 'yas_bounds': (-8.0, 96.0), 
 'customer_harcama_outliers_count': 0, 'harcama_bounds': (-2407.415, 7514.085), 
 'sales_fiyat_outliers_count': 5, 'fiyat_bounds': (-759.5137500000001, 2242.41625),             #outliers are here (up to upp_limit)
 'sales_toplam_outliers_count': 37, 'toplam_bounds': (-12068.63625, 25779.03375),               #outliers are here (up to upp_limit)
 'sales_adet_outliers_count': 0, 'adet_bounds': (-10.0, 30.0)}
"""

    ##Equalizing the upp_limit to use outliers without corrupting the dataset 
sales_df["toplam_satis"] = sales_df["toplam_satis"].apply(
    lambda x: min(max(x, toplam_low), toplam_upp))
sales_df["fiyat"] = sales_df["fiyat"].apply(
    lambda x: min(max(x, fiyat_low), fiyat_upp))

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

#new outliers check
#print("""
#      -------------------------------------
#        New Outliers Check:
#      -------------------------------------
#      """)
#n_outliers_t = merged_df[(merged_df["toplam_satis"] < toplam_low) | (merged_df["toplam_satis"] > toplam_upp)]
#print(n_outliers_t)
#n_outliers_f = merged_df[(merged_df["fiyat"] < fiyat_low) | (merged_df["fiyat"] > fiyat_upp)]
#print(n_outliers_f)
#
#print("""
#      -------------------------------------
#        New Nan Check:
#      -------------------------------------
#      """)
#print(merged_df.isnull().sum())
#for x in merged_df["toplam_satis"]:
#    if x == None:
#        print(x)
#for y in merged_df["fiyat"]:
#    if y == None:
#        print(x)
#


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
['Elektronik' 'Kırtasiye' 'Ev Aletleri' 'Giyim' 'Mutfak Ürünleri']
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
        Weekly Total Item Sales (First 5):
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


####Mission 3: Categorical and Numerical Analysis
### Category Analysis

###3.1: Categoric Total Sales
categoric_sales = merged_df.groupby(["kategori"])
categoric_sales_sum =categoric_sales["toplam_satis"].sum()

#print(categoric_sales_sum)

print("""
      -------------------------------------
        Proportion of Sales (%):
      -------------------------------------
      """)
print((categoric_sales_sum / categoric_sales_sum.sum())*100)
"""(print)
kategori
Elektronik         48.496411
Ev Aletleri        10.114124
Giyim              10.353281
Kırtasiye          21.005956
Mutfak Ürünleri    10.030227
"""


###3.2: Age Group Analysis
merged_df["age_group"] = pd.cut(merged_df["yas"], bins=[0, 25, 35, 50, 100], labels=["18-25", "26-35", "36-50", "50+"])

age_group_sales = merged_df.groupby("age_group", observed=False)["toplam_satis"].sum()
print("""
      -------------------------------------
        Sales Trends by Age Groups:
      -------------------------------------
      """)
print((age_group_sales / age_group_sales.sum())*100)


age_group_item_sales = merged_df.groupby(["age_group", "kategori"], observed=False)["toplam_satis"].sum()
print("""
      -------------------------------------
        Categoric Sales Trends by Age Groups:
      -------------------------------------
      """)
print((age_group_item_sales / age_group_item_sales.sum())*100)

"""(print)
Categoric Sales Trends by Age Groups:
18-25      Elektronik          7.714356     #max
           Ev Aletleri         1.669860
           Giyim               1.699330
           Kırtasiye           3.102833
           Mutfak Ürünleri     1.840506
26-35      Elektronik          8.988815     #max
           Ev Aletleri         1.848845
           Giyim               1.608566
           Kırtasiye           3.831070
           Mutfak Ürünleri     1.804537
36-50      Elektronik         14.131741     #max
           Ev Aletleri         3.022600
           Giyim               3.586270
           Kırtasiye           6.398622
           Mutfak Ürünleri     2.880272
50+        Elektronik         17.661499     #max
           Ev Aletleri         3.572819
           Giyim               3.459116
           Kırtasiye           7.673431
           Mutfak Ürünleri     3.504912
Name: toplam_satis, dtype: float64
"""

"""(print) (after Elektronik and Kırtasiye)
18-25   Mutfak Ürünleri
26-35   Ev Aletleri
36-50   Giyim
50+     Ev Aletleri
"""


###3.3: Gender Analysis

gender_group_sales = merged_df.groupby("cinsiyet")["toplam_satis"].sum()
print("""
      -------------------------------------
        Sales Trends by Gender Groups:
      -------------------------------------
      """)
print((gender_group_sales / gender_group_sales.sum())*100)


gender_group_item_sales = merged_df.groupby(["cinsiyet", "kategori"])["toplam_satis"].sum()
print("""
      -------------------------------------
        Categoric Sales Trends by Gender Groups:
      -------------------------------------
      """)
print((gender_group_item_sales / gender_group_item_sales.sum())*100)

"""(print)
Categoric Sales Trends by Gender Groups:
cinsiyet  kategori
Erkek     Elektronik         23.236951      #max
          Ev Aletleri         4.693817
          Giyim               5.656780
          Kırtasiye          11.028295
          Mutfak Ürünleri     5.015022
Kadın     Elektronik         25.259459      #max
          Ev Aletleri         5.420307
          Giyim               4.696502
          Kırtasiye           9.977661
          Mutfak Ürünleri     5.015205
Name: toplam_satis, dtype: float64
"""

"""(print) (after Elektronik and Kırtasiye)
Erkek   Giyim
Kadın   Ev Aletleri
"""


####Mission 4: Advanced Data Manipulation

###4.1.1: City Based Analysis
city_group_sales = merged_df.groupby("sehir")["harcama_miktari"].sum().sort_values(ascending=False)

print("""
      -------------------------------------
        Spending by City:
      -------------------------------------
      """)
print(city_group_sales)

###4.1.2: Customer-City Analysis
customer_city_group_sales = merged_df.groupby(["musteri_id", "sehir"])["harcama_miktari"].sum()
id_customer_city_group_sales = customer_city_group_sales.groupby("sehir").idxmax()
max_customer_city_group_sales = customer_city_group_sales.groupby("sehir").max().sort_values(ascending=False)

customer_city_sales = pd.DataFrame({
    "Customer İd": [idx[0] for idx in id_customer_city_group_sales],
    "Amount of Expenditure": max_customer_city_group_sales
})

print("""
      -------------------------------------
        Maximum Amount Spending by Customer ID and City:
      -------------------------------------
      """)
print(customer_city_sales)

"""(print)
           Customer İd  Amount of Expenditure
sehir
Gaziantep         8128               23039.85
Ankara            4342               22392.70
Bursa             1522               19933.92
Konya             6095               19473.60
Adana             8939               19361.36
İstanbul          3175               19186.56
Antalya           6869               19183.00
İzmir             8347               18778.84
"""

###4.2: Calculate the Average Sales Growth Rate for Each Product 

product_monthly_sales = merged_df.groupby([merged_df["tarih"].dt.to_period("M"), "ürün_adi"])["adet"].sum()
product_monthly_sales_ch = product_monthly_sales.groupby("ürün_adi").pct_change() * 100
product_average_sales_gr = product_monthly_sales_ch.groupby("ürün_adi").mean()

print("""
      -------------------------------------
        Average Sales Growth Rate for Each Product:
      -------------------------------------
      """)
print(product_average_sales_gr)

"""(print)
ürün_adi
Bilgisayar    16.172751
Defter         9.207474
Fırın          5.443718
Kalem         11.632030
Klima          2.130580
Kulaklık       6.577236
Mouse          4.523185
Su Şişesi      2.097570
Telefon       -2.015909
Çanta          4.617467
Name: adet, dtype: float64
"""


###4.3: Monthly Total Sales and Change Analysis by Category

category_monthly_sales = merged_df.groupby([merged_df["tarih"].dt.to_period("M"), "kategori"])["toplam_satis"].sum()
category_monthly_sales_ch = category_monthly_sales.groupby("kategori").pct_change() * 100
category_monthly_sales_ch = category_monthly_sales_ch.reset_index()
category_monthly_sales_ri = category_monthly_sales.reset_index()

    # Graphs (monthly_total_sales_of_each_category)
#plt.figure(figsize=(14, 8))
#for kategori in category_monthly_sales_ri["kategori"].unique():
#    kategori_veri = category_monthly_sales_ri[category_monthly_sales_ri["kategori"] == kategori]
#    plt.plot(kategori_veri["tarih"].astype(str), kategori_veri["toplam_satis"], marker="o", label=kategori)
#plt.title("Monthly Category Sales")
#plt.xlabel("Date")
#plt.ylabel("Category Sales")
#plt.xticks(rotation=45)
#plt.legend(title="Kategori")
#plt.grid(True)
#plt.tight_layout()
#plt.show()

    # Graphs (monthly_total_sales_change_ratesof_each_category)
#plt.figure(figsize=(14, 8))
#for kategori in category_monthly_sales_ch["kategori"].unique():
#    kategori_veri = category_monthly_sales_ch[category_monthly_sales_ch["kategori"] == kategori]
#    plt.plot(kategori_veri["tarih"].astype(str), kategori_veri["toplam_satis"], marker="o", label=kategori)
#
#plt.title("Monthly Total Sales Change Rates of Each Category")
#plt.xlabel("Tarih")
#plt.ylabel("Değişim Oranı (%)")
#plt.xticks(rotation=45)
#plt.legend(title="Kategori")
#plt.grid(True)
#plt.tight_layout()
#plt.show()


####Mission 5: Extra(Bonus)

###5.1: Pareto Analysis (80/20)
product_sales = merged_df.groupby("ürün_adi")["toplam_satis"].sum().sort_values(ascending=False)
total_sales = product_sales.sum()
pareto_limit = total_sales * 0.8
total = 0
for product, sales in product_sales.items():
    total += sales
    if total >= pareto_limit:
        break


product_sales_pareto = product_sales[product_sales.cumsum() <= pareto_limit]

#print("""
#      -------------------------------------
#        Pareto Analysis: Products that make up the top %80 of sales:
#      -------------------------------------
#      """)
#for product in product_sales_pareto.index:
#    print(product)
    
"""(print)
Kalem
Telefon
Çanta
Defter
Fırın
Su Şişesi
Mouse
"""

    #Graphs (product_sales_pareto)
#plt.figure(figsize=(12, 6))
#plt.plot(product_sales_pareto.index, product_sales_pareto.values, marker="o",label="product_sales_pareto")
#plt.title("Product Sales Pareto")
#plt.xlabel("Date")
#plt.ylabel("Sales")
#plt.xticks(rotation=45)
#plt.legend()
#plt.grid(True)
#plt.show()


###5.2: Cohort Analysis
    # Find customers' first purchase date
customers_first = merged_df.groupby("musteri_id")["tarih"].min().reset_index()
customers_first.columns = ["musteri_id", "first_sale_date"]

merged_df_first_sale = pd.merge(merged_df, customers_first, on="musteri_id", how="inner")
#merged_df_first_sale.to_csv("datasets\\custom\\merged_data_w_first_sale.csv", index=False)

    # Create cohort month (based on first purchase date)
merged_df_first_sale["cohort_month"] = merged_df_first_sale["first_sale_date"].dt.to_period("M")

    # Create purchase month (based on sales date)
merged_df_first_sale["sale_month"] = merged_df_first_sale["tarih"].dt.to_period("M")

    # Calculate number of users for cohort analysis
cohort_data = merged_df_first_sale.groupby(["cohort_month", "sale_month"]).agg({"musteri_id": pd.Series.nunique}).reset_index()
cohort_data.columns = ["cohort_month", "sale_month", "active_customer_count"]

    # Calculate how many months have passed for each cohort
cohort_data["passed_months"] = (cohort_data["sale_month"] - cohort_data["cohort_month"]).apply(lambda x: x.n)

    # Cohort Table
cohort_pivot = cohort_data.pivot_table(index="cohort_month", columns="passed_months", values="active_customer_count")

    # Calculate the purchase rate for each month, based on the first month
cohort_pivot_percentage = cohort_pivot.divide(cohort_pivot.iloc[:, 0], axis=0) * 100

    # Results
#print("Cohort Analysis (% Uptake Rate):")
#print(cohort_pivot_percentage)

    # Graphs (cohort_analysis_customer_repurchase_rate)
#plt.figure(figsize=(14, 10))
#sns.heatmap(cohort_pivot_percentage, annot=True, fmt=".1f", cmap="YlGnBu")
#plt.title("Cohort Analysis: Customer Repurchase Rate (%)")
#plt.xlabel("Passed Months")
#plt.ylabel("Cohort Months")
#plt.show()



###5.3: Basic Regression Model (Weekly)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

   ## Calculate total weekly sales
weekly_sales_md = merged_df.resample("W", on="tarih")["toplam_satis"].sum().reset_index()

   ## Set features (weeks) and target variable (sales amount)
weekly_sales_md["week"] = np.arange(len(weekly_sales_md))    # Represent weeks numerically
X = weekly_sales_md[["week"]]                                # Independent variable
y = weekly_sales_md["toplam_satis"]                          # Dependent variable

   ## Separating data as train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   ## Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

   ## Make the model's predictions
y_pred = model.predict(X_test)

   ## Measuring model accuracy
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

   ## Results
#print("""
#      -------------------------------------
#        Model Errors:
#      -------------------------------------
#      """)
#print("Mean Absolute Error (MAE):", mae)
#print("Mean Squared Error (MSE):", mse)
#print("Root Mean Squared Error (RMSE):", rmse)
#print("R^2 Score:", r2)

    # Graphs (basic_reg_weekly_predict)
#plt.figure(figsize=(14, 6))
#plt.plot(X_test["week"], y_test, "o", label="Real Sales", markersize=8)
#plt.plot(X_test["week"], y_pred, "r-", label="Predict Sales")
#plt.xlabel("Week")
#plt.ylabel("Total Sales")
#plt.title("Actual and Estimated Weekly Sales Amounts")
#plt.legend()
#plt.grid(True)
#plt.show()


###5.4: RFM (Recency, Frequency, Monetary) Analysis

    # Set a reference date for analysis (for example, one day after the latest date of the data set)
referance_date = merged_df_first_sale["tarih"].max() + pd.Timedelta(days=1)

    # Calculate RFM metrics
rfm = merged_df_first_sale.groupby("musteri_id").agg({
    "tarih": lambda x: (referance_date - x.max()).days,   # Recency
    "musteri_id": "count",                                # Frequency
    "toplam_satis": "sum"                                 # Monetary
})

    # Change column names
rfm.columns = ["Recency", "Frequency", "Monetary"]

    # Filter customers with a Monetary value of 0
rfm = rfm[rfm["Monetary"] > 0]

    # Determine RFM scores
rfm["R_Score"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1])
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4])
rfm["M_Score"] = pd.qcut(rfm["Monetary"], 4, labels=[1, 2, 3, 4])

    # Combining RFM Score
rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)

    # Segments
segment_map = {
    "444": "Champions",
    "344": "Loyal Customers",
    "144": "Potential Loyalists",
    "244": "New Customers",
    "441": "At Risk",
    "111": "Hibernating"
}
rfm["Segment"] = rfm["RFM_Score"].map(segment_map).fillna("Others")

    # Results
#print("""
#      -------------------------------------
#        RFM Analysis:
#      -------------------------------------
#      """)
#print(rfm.head(15))

"""(print (15))
      -------------------------------------
        RFM Analysis:
      -------------------------------------
            Recency  Frequency  Monetary R_Score F_Score M_Score RFM_Score        Segment
musteri_id
1004            224          2  13811.82       3       3       3       333         Others
1006            140          1  14890.10       3       1       3       313         Others
1007            295          1   4135.29       2       1       2       212         Others
1009            171          2  11625.38       3       3       3       333         Others
1012            169          1   5463.05       3       1       2       312         Others
1013            563          2   7648.86       1       3       2       132         Others
1015            154          1  13303.01       3       1       3       313         Others
1018            450          3  36361.28       2       4       4       244  New Customers
1020            293          2   9358.74       2       3       2       232         Others
1024            585          1   2756.35       1       1       1       111    Hibernating
1029            100          1   2977.20       4       1       1       411         Others
1033            108          3  21208.30       4       4       4       444      Champions
1034             71          2   5650.14       4       3       2       432         Others
1040            708          1  12854.93       1       1       3       113         Others
1043            646          1  11116.71       1       1       3       113         Others
"""

    #Frequent customers
frequent_customers = rfm[rfm["Segment"] == "Loyal Customers"]
#print("""
#      -------------------------------------
#        RFM Analysis- Loyal Customers:
#      -------------------------------------
#      """)
#print(frequent_customers.head())

"""(print)
      -------------------------------------
        RFM Analysis- Loyal Customers:
      -------------------------------------

            Recency  Frequency  Monetary R_Score F_Score M_Score RFM_Score          Segment
musteri_id
1116            173          3  47421.10       3       4       4       344  Loyal Customers
1150            163          4  23606.59       3       4       4       344  Loyal Customers
1274            137          3  17617.06       3       4       4       344  Loyal Customers
1324            268          3  26002.39       3       4       4       344  Loyal Customers
1522            179          4  26099.93       3       4       4       344  Loyal Customers
"""