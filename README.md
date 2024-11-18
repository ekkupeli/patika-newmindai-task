# Data Analysis Project

## Project Overview
This project was assigned as part of the Patika & NewMind AI Bootcamp. The main objective of this project is to practice data analysis by processing and analyzing customer and sales data sets created with randomly generated values.

The project covers the following key areas:
- **Data Cleaning and Processing**
- **Time Series Analysis**
- **Categorical and Numerical Analysis**
- **Advanced Data Processing**
- **Data Visualization**
- **Bonus Section**

## Technology and Environment Setup
The project was coded using the Python programming language. The required libraries for this project are listed in the 'env' folder:
- If you are using **Anaconda**, the `environment.yml` file can be used to create the necessary environment.
- For non-Anaconda users, the `environment_fh.yml` file can be used for relevant library versions.

## Data Sets
The data sets utilized in this project are located in the `datasets/base` folder. These include:
- **Customer Data**
- **Sales Data**

Both data sets have been prepared to simulate real-world data for the purpose of analysis.

## Task Document
The task details and requirements are documented in a `.docx` file, which is available in the project directory.


### Task 1: Data Cleaning and Processing

In this section, two datasets—`sales_df` (sales data) and `customer_df` (customer data)—were examined. The data cleaning and processing steps were carried out as follows:

#### **1.1. Data and Missing Value Check**

First, both datasets were loaded from CSV files, and the first few rows of the data were displayed:

- **Sales Data**: `sales_df`
- **Customer Data**: `customer_df`

The general structure and column information of each dataset were checked using the `.info()` function. Additionally, the frequency distribution of specific columns in each dataset, such as `ürün_adi`, `kategori` for the sales data, and `cinsiyet`, `yas` and `sehir`  for the customer data, was examined using `.value_counts()`. This step provided an overview of the distribution of categorical variables in both datasets.

#### **1.2. Outlier Detection**

The **Interquartile Range (IQR)** method was used for outlier detection. This method identifies outliers by measuring the spread of the middle 50% of the data and flagging values that fall outside this range.

Outliers were searched in the following columns:
- **Customer Age**: `yas`
- **Customer Spending**: `harcama_miktari`
- **Sales Price**: `fiyat`
- **Total Sales**: `toplam_satis`
- **Sales Quantity**: `adet`

For each of these columns, the **lower and upper limits** (`low_limit`, `upp_limit`) were calculated, and outliers were flagged based on these bounds. The count of outliers and the corresponding limits were summarized. 

To prevent the outliers from distorting the dataset, the outliers in the `toplam_satis` column were adjusted to the upper limit. This step ensured that the data remained intact without the influence of extreme outliers.

#### **1.3. Merging Data**

Finally, the **Sales Data** and **Customer Data** were merged using the `"musteri_id"` column. This ensured that each sales record was associated with the corresponding customer information.

The merged dataset, `merged_df`, was created, and sample data was displayed. Also saved it as `merged_data.csv` in `datasets/custom` folder.

