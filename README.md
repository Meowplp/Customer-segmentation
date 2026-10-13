# Customer Segmentation Using K-Means Clustering

**Unsupervised Machine Learning | Customer Analytics | Python | PCA | K-Means**

## Project Overview

This project applies **unsupervised machine learning** to segment customers of a grocery retailer based on their demographic characteristics, purchasing behavior, and engagement with marketing campaigns.

Using a dataset of **2,240 customers and 29 features**, the analysis explores how income, household composition, spending patterns, and purchasing channels influence customer behavior.

The objective is to identify distinct customer groups and translate these findings into actionable insights that support **targeted marketing, customer engagement, and resource allocation**.

**Dataset:** [Marketing Campaign Dataset – Kaggle](https://www.kaggle.com/datasets/jackdaoud/marketing-data)

## Business Problem

Customers differ in their purchasing power, shopping preferences, and responsiveness to promotional campaigns. Applying the same marketing strategy to all customers may lead to inefficient marketing expenditure and missed opportunities.

This project addresses the following question:

**How can customers be segmented into meaningful groups based on their demographics and purchasing behavior to support more effective marketing strategies?**

The analysis focuses on:

- Identifying customer groups with similar demographic and behavioral characteristics.
- Understanding differences in spending, purchasing channels, and promotional responsiveness.
- Developing interpretable customer profiles.
- Recommending targeted marketing strategies for each segment.

## Dataset Description

The dataset contains 2,240 customer records across 29 variables, covering four main areas.

| Category | Key Features | Description |
|---|---|---|
| Demographics | `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome` | Customer background, income, and household composition |
| Purchasing Behavior | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` | Spending across product categories over the previous two years |
| Shopping Channels | `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth` | Purchasing activity across online, catalog, and physical stores |
| Marketing Engagement | `AcceptedCmp1`–`AcceptedCmp5`, `Response`, `NumDealsPurchases` | Campaign acceptance and discount-related purchasing behavior |

Additional variables include `Recency`, representing days since the customer's last purchase, and `Dt_Customer`, indicating their enrollment date.

## Analytical Methodology

The project follows a five-stage analytical workflow:

1. **Exploratory Data Analysis (EDA):** Examine data distributions, customer characteristics, and relationships between variables.
2. **Data Cleaning and Feature Engineering:** Handle missing values, consolidate categorical variables, identify outliers, and create behavioral features.
3. **Feature Scaling and Dimensionality Reduction:** Standardize features and apply Principal Component Analysis (PCA).
4. **Customer Clustering:** Use the Elbow Method to determine the number of clusters and apply K-Means.
5. **Customer Profiling and Business Insights:** Interpret clusters based on spending, demographics, and purchasing behavior.

## 1. Data Cleaning and Preparation

### 1.1 Handling Missing Values

An initial data quality assessment identified **24 missing values in the `Income` column**, representing approximately 1.1% of the dataset. No missing values were detected in the remaining columns.

Rather than removing these customer records, missing income values were replaced with the column mean.

**Method:** Mean imputation

**Rationale:** Preserve customer records for clustering while maintaining a complete numerical dataset.

### 1.2 Cleaning Categorical Variables

The dataset contained multiple education and marital-status categories. These were consolidated into broader groups to simplify interpretation and reduce categorical complexity.

**Education categories**

| Original Values | Consolidated Category |
|---|---|
| Basic | Undergraduate |
| Graduation | Graduate |
| Master, PhD, 2n Cycle | Postgraduate |

**Marital-status categories**

| Original Values | Consolidated Category |
|---|---|
| Married, Together | Married |
| Single, Divorced, Widow, Alone | Unmarried |
| Absurd, YOLO | Removed |

Four records containing ambiguous marital-status labels (`Absurd` and `YOLO`) were excluded.

The consolidated categorical variables were subsequently transformed into numerical representations using `OrdinalEncoder` from Scikit-learn.

### 1.3 Feature Engineering

Several new features were constructed to capture more meaningful customer characteristics.

| Engineered Feature | Calculation | Analytical Purpose |
|---|---|---|
| `Age` | 2024 − `Year_Birth` | Understand customer age distribution |
| `TotalSpent` | Sum of spending across six product categories | Measure overall customer spending |
| `FamilySize` | Estimated adults + `Kidhome` + `Teenhome` | Understand household composition |
| `Is_Parent` | 1 if the household has children or teenagers; otherwise 0 | Differentiate households with and without children |
| `Total_promos` | Sum of `AcceptedCmp1` through `AcceptedCmp5` | Measure historical campaign acceptance |

These engineered variables provide a more interpretable representation of customers for subsequent exploratory analysis and profiling.

### 1.4 Outlier Detection and Treatment

Outliers were investigated using box plots across numerical features, including age, income, spending, household size, and purchasing frequency.

![Box plots for numerical feature distributions](https://github.com/user-attachments/assets/41e31489-07db-443a-9059-19f2694cbfde)

The **Interquartile Range (IQR)** method was applied to identify and remove observations outside the expected range.

The lower and upper thresholds were calculated as:

- Lower bound = Q1 − 1.5 × IQR
- Upper bound = Q3 + 1.5 × IQR
- IQR = Q3 − Q1

This method was used to reduce the influence of extreme values on distance-based clustering.

Following categorical cleaning and outlier removal, the dataset contained **1,512 customer records**, approximately 67.5% of the original dataset.

### 1.5 Removing Unnecessary Features

Variables that were redundant or not required for clustering were removed, including:

- `ID`: Unique customer identifier.
- `Year_Birth`: Replaced by the engineered `Age` feature.
- `Adult`: Intermediate variable used to calculate household size.
- `Z_CostContact` and `Z_Revenue`: Excluded because they provided limited useful variation for this analysis.

## 2. Exploratory Data Analysis

Exploratory analysis was conducted to understand customer demographics, spending patterns, and purchasing preferences.

**Key observations:**

- **Income and spending:** Customer income showed a positive relationship with total expenditure.
- **Product preferences:** Wine accounted for the largest share of spending, followed by meat products.
- **Household composition:** Smaller households tended to spend more on wine and gold products than larger households.
- **Purchasing channels:** In-store purchasing was a prominent shopping channel across customer groups.
- **Customer behavior:** Income, total spending, and catalog purchases showed notable correlations.

These findings informed the interpretation of customer segments after clustering.

## 3. Feature Scaling and Dimensionality Reduction

### 3.1 Feature Standardization

Since the dataset included variables measured on different scales, such as income, age, and purchase frequency, `StandardScaler` was applied before clustering.

Standardization transforms features to have approximately zero mean and unit variance, reducing the dominance of variables with larger numerical ranges.

### 3.2 Principal Component Analysis (PCA)

**Principal Component Analysis (PCA)** was used to reduce dimensionality while retaining a substantial proportion of the variation in customer characteristics.

The process involved:

1. Applying PCA with 10 components to inspect explained variance.
2. Evaluating the contribution of individual principal components.
3. Selecting four principal components, which explained approximately **70% of total variance**.
4. Using the reduced feature space as input for K-Means clustering.

This approach reduced feature complexity and provided a lower-dimensional representation for identifying customer groups.

## 4. Customer Segmentation Using K-Means

### 4.1 Selecting the Number of Clusters

The **Elbow Method** was used to evaluate clustering performance across different values of K.

Within-cluster sum of squares (WCSS), also known as inertia, measures how closely observations are grouped around their respective centroids.

Based on the observed trade-off between the number of clusters and declining inertia, **K = 5** was selected.

### 4.2 K-Means Implementation

K-Means clustering was performed on the four principal components using Scikit-learn.

The model assigned each customer to one of five mutually exclusive clusters based on similarity in the reduced feature space.

The resulting clusters were visualized in two dimensions to explore their relative distribution.

![K-Means customer clusters](https://github.com/user-attachments/assets/8888602b-8ab2-4feb-9862-e902ddfbc891)

### 4.3 Income and Spending Analysis

Income and total spending were examined across the five clusters to identify differences in customer value and purchasing power.

![Income and spending by customer cluster](https://github.com/user-attachments/assets/3f727199-be81-404e-babd-c1c3d7a15675)

The analysis revealed clear differences between customer groups:

- Clusters 1 and 3 demonstrated relatively high income and spending.
- Cluster 2 represented lower-income, lower-spending customers.
- Clusters 0 and 4 represented intermediate spending groups with different demographic and purchasing characteristics.

## 5. Customer Segment Profiles

Five customer segments were identified and interpreted using demographic characteristics, shopping behavior, and campaign engagement.

![Customer segmentation summary](https://github.com/user-attachments/assets/27d93783-d1e6-4b6b-baac-e5c0b789021d)

| Cluster | Customer Segment | Key Characteristics |
|---|---|---|
| **0** | Moderate-Spending Small Households | Low-to-middle income, moderate spending, smaller households, preference for online and in-store purchases |
| **1** | High-Value Traditional Shoppers | Higher income and spending, strong catalog and in-store purchasing activity, historical campaign engagement |
| **2** | Budget-Conscious Family Shoppers | Lower income and spending, larger households, limited purchasing activity across channels |
| **3** | High-Value Deal Seekers | Upper-middle to high income, high spending, preference for online and in-store shopping, relatively strong discount-purchasing activity |
| **4** | Moderate-Spending Family Shoppers | Middle-to-upper income, moderate spending, larger households, preference for online and in-store channels |

### Cluster 0 — Moderate-Spending Small Households

Customers in this segment typically have low-to-middle incomes, generally below $60,000 annually, and moderate spending levels.

They tend to belong to smaller households of one or two people, with ages primarily ranging from the late 20s to under 60.

**Purchasing behavior:** Prefer website and in-store purchases over catalog shopping.

**Marketing opportunity:** Use personalized online promotions and cross-selling campaigns to encourage higher spending and purchase frequency.

### Cluster 1 — High-Value Traditional Shoppers

This segment consists of relatively high-income customers with high overall spending.

Customers are generally between 40 and 80 years old, live in smaller households, and have graduate or postgraduate education levels.

**Purchasing behavior:** Highest catalog and in-store purchasing activity among the five segments, with some customers accepting multiple previous marketing offers.

**Marketing opportunity:** Prioritize retention, loyalty programs, premium product recommendations, and personalized catalog campaigns.

### Cluster 2 — Budget-Conscious Family Shoppers

This segment is characterized by lower incomes, generally below $50,000 annually, and the lowest overall spending.

Customers are typically between 40 and 80 years old, with households containing two to four people.

**Purchasing behavior:** Lowest purchasing activity across web, catalog, and in-store channels, with particularly limited catalog purchases.

**Marketing opportunity:** Explore affordable product bundles, value-focused promotions, and targeted offers designed to encourage purchasing activity.

### Cluster 3 — High-Value Deal Seekers

Customers in this segment have upper-middle to high incomes and relatively high spending.

They are generally adults below 80 years old, with households of two to three people, and have graduate or postgraduate education levels.

**Purchasing behavior:** Strong online and in-store purchasing activity, alongside relatively high participation in discount-related purchases.

**Marketing opportunity:** Use targeted promotional offers, personalized vouchers, and online campaigns to encourage repeat purchases.

### Cluster 4 — Moderate-Spending Family Shoppers

This segment consists of customers with middle-to-upper income levels and moderate spending.

Customers are typically middle-aged or older, living in households of two to four people.

**Purchasing behavior:** Prefer online and in-store shopping, although their purchase frequency is lower than that of Cluster 3.

**Marketing opportunity:** Introduce household-oriented bundles, personalized recommendations, and cross-selling initiatives to increase average spending.

## 6. Business Recommendations

The customer segmentation results suggest that marketing strategies should be differentiated according to customer value, purchasing channels, and promotional responsiveness.

| Strategic Priority | Target Segments | Recommended Action |
|---|---|---|
| Customer retention | Clusters 1 and 3 | Prioritize loyalty programs and personalized engagement for high-spending customers |
| Promotional optimization | Cluster 3 | Test targeted discounts and digital vouchers to encourage repeat purchases |
| Customer activation | Cluster 2 | Explore value-oriented campaigns and affordable product bundles |
| Cross-selling | Clusters 0 and 4 | Recommend complementary products and household-specific offers |
| Channel personalization | All clusters | Tailor campaign delivery based on observed preferences for web, catalog, and in-store shopping |

These recommendations represent potential marketing applications of the clustering findings. Their effectiveness would need to be validated through subsequent campaign testing.

## 7. Key Findings and Conclusion

The analysis identified **five distinct customer segments** with meaningful differences in income, spending, household composition, and purchasing behavior.

Three important business insights emerged:

**1. High-spending customers are not a homogeneous group.**

Clusters 1 and 3 both demonstrate high spending, but their purchasing behavior differs. Cluster 1 shows stronger catalog and in-store activity, while Cluster 3 exhibits stronger online purchasing and discount-related behavior.

**2. Household composition provides useful context for customer segmentation.**

Smaller households are more prevalent among certain high-spending segments, while larger households are more common among lower- and moderate-spending groups.

**3. Marketing strategies can be tailored beyond income alone.**

Purchasing channels, promotional engagement, and family size provide additional information for designing segment-specific campaigns.

Overall, this project demonstrates how **data preprocessing, feature engineering, dimensionality reduction, and unsupervised learning** can transform customer transaction and demographic data into interpretable segments that support marketing decision-making.

## 8. Tools and Technologies

| Technology | Application |
|---|---|
| Python | Data analysis and modeling |
| Pandas, NumPy | Data manipulation, cleaning, and feature engineering |
| Matplotlib, Seaborn | Exploratory analysis and visualization |
| Scikit-learn | Feature encoding, standardization, PCA, and K-Means clustering |
| Jupyter Notebook | Analytical development and documentation |

**Machine Learning Techniques:** Mean Imputation · IQR Outlier Detection · Ordinal Encoding · StandardScaler · PCA · Elbow Method · K-Means Clustering

## 9. Limitations and Future Improvements

Several considerations could improve the reliability and practical applicability of the analysis.

- **Outlier treatment:** IQR-based filtering reduced the dataset from 2,240 to 1,512 observations. Future analysis could compare clustering results with more conservative outlier treatment to avoid excluding legitimate high-value customers.
- **Categorical encoding:** Ordinal encoding may introduce artificial numerical relationships between categories. One-hot encoding or alternative approaches could be evaluated.
- **Cluster validation:** Silhouette scores and other clustering metrics could supplement the Elbow Method to assess cluster separation and stability.
- **Feature selection:** Further experiments could evaluate the impact of correlated variables and alternative feature sets.
- **Business validation:** Targeted marketing experiments could assess whether segment-specific strategies improve engagement, conversion, and customer spending.

## Data Source

[Marketing Campaign Dataset — Kaggle](https://www.kaggle.com/datasets/jackdaoud/marketing-data)

The dataset is used for educational and portfolio purposes.
