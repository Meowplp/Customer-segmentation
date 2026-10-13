# Customer Segmentation Using K-Means Clustering

**Python | Data Cleaning | Exploratory Data Analysis | PCA | K-Means | Customer Analytics**

## Project Overview

This project applies **unsupervised machine learning** to identify customer segments based on demographic characteristics, purchasing behavior, and marketing campaign engagement.

Using a grocery retailer's marketing dataset containing **2,240 customers and 29 features**, the analysis aims to understand customer spending patterns and identify opportunities for personalized marketing strategies.

**Business Question:** How can customers be segmented into meaningful groups to improve targeted marketing and customer engagement?

**Dataset:** [Marketing Campaign Dataset – Kaggle](https://www.kaggle.com/datasets/jackdaoud/marketing-data)

## EDA & Data Cleaning
### 1. Data Cleaning & Feature Engineering

The dataset was prepared through the following steps:

| Step | Method | Purpose |
|---|---|---|
| Missing values | Mean imputation for 24 missing `Income` values | Preserve customer records |
| Categorical cleaning | Consolidated education and marital-status categories | Simplify customer demographics |
| Categorical encoding | OrdinalEncoder | Convert categorical variables into numerical values |
| Feature engineering | Created `Age`, `TotalSpent`, `FamilySize`, and `Is_Parent` | Capture meaningful customer characteristics |
| Outlier treatment | Interquartile Range (IQR) | Reduce the influence of extreme observations |
| Feature selection | Removed identifiers and redundant variables | Reduce unnecessary features |

After preprocessing, **1,512 customer records** were retained for clustering.

#### Outlier Detection

Box plots were used to examine numerical distributions and identify potential outliers before applying IQR-based filtering.

![Outlier Detection](https://github.com/user-attachments/assets/41e31489-07db-443a-9059-19f2694cbfde)

### 2. Exploratory Data Analysis

Exploratory analysis investigated customer demographics, household composition, spending patterns, and relationships between variables.

**Key findings:**

- Customer ages are concentrated around 45–60 years.
- Three-person households represent the largest customer group.
- Wine contributes the largest share of product spending, followed by meat.
- Income and total spending show a strong positive correlation (r ≈ 0.77).
- Total spending is also strongly associated with catalog and in-store purchases.

**Additional visualizations:** Customer age distribution, spending by family size, and correlation heatmap.

## Dimensionality Reduction & K-Means Clustering

### Principal Component Analysis (PCA)

Before clustering, `StandardScaler` was applied to standardize the dataset. PCA was then used to reduce dimensionality and capture the main patterns of customer variation.

Four principal components were selected, explaining approximately **51% of the total variance**.

### K-Means Clustering

The Elbow Method was used to evaluate different numbers of clusters based on within-cluster sum of squares (WCSS).

**K = 5** was selected to balance cluster interpretability and model complexity.

![K-Means Clustering](https://github.com/user-attachments/assets/8888602b-8ab2-4feb-9862-e902ddfbc891)

The resulting clusters show differences in purchasing power, spending behavior, household composition, and shopping-channel preferences.

## Customer Segmentation Results

### Income and Spending Analysis

![Income and Spending Clusters](https://github.com/user-attachments/assets/3f727199-be81-404e-babd-c1c3d7a15675)

Five customer segments were identified:

| Cluster | Customer Segment | Key Characteristics |
|---|---|---|
| **0** | Moderate-Spending Small Households | Low-to-middle income, moderate spending, smaller households, preference for web and in-store shopping |
| **1** | High-Value Traditional Shoppers | High income and spending, strongest catalog purchasing activity, high in-store activity |
| **2** | Budget-Conscious Family Shoppers | Lower income and spending, larger households, lowest purchasing activity |
| **3** | High-Value Multichannel Shoppers | Higher income and spending, strongest web and in-store purchasing activity, frequent deal purchases |
| **4** | Moderate-Spending Family Shoppers | Middle-income customers, moderate spending, larger households, preference for web and in-store shopping |

### Purchasing Behavior & Marketing Engagement

Analysis of purchasing channels and historical marketing responses revealed that:

- **Cluster 1** has the highest average catalog purchases and strong in-store activity.
- **Cluster 3** leads in web and in-store purchases and shows relatively frequent discount-related purchasing.
- **Cluster 2** has the lowest purchasing activity despite relatively frequent website visits.
- Most customers across all clusters have not accepted previous marketing campaigns.
- Higher-income customers are not necessarily homogeneous in their channel preferences or promotional behavior.

### Demographic Differences

Age, family size, and education were also examined to develop interpretable customer profiles.

<img width="1618" height="551" alt="image" src="https://github.com/user-attachments/assets/4a81bc2b-b7b5-49f4-85d6-a99492b4e84c" />
Cluster 1 and Cluster 3 only contain customers of Graduate and Postgraduate levels while other clusters contains all three levels. Since the number of customers in Postgraduate level outweight the other groups, so the education factor might not be a good characteristic for this store's marketing campaign design
<img width="595" height="584" alt="image" src="https://github.com/user-attachments/assets/9bdc8cda-195b-49d6-a7f8-98fe38a2d8f4" />
Customers in cluster 0 have age range from above 20 to under 60
Customers in cluster 1 have age range from 40 to under 80
Customer in cluster 2 have age range from around 40 to 80
Customer in cluster 3 have age range from under 40 to just 80
Customer in cluster 4 have age range around 40 to above 80


## 5. Customer Profiles & Business Recommendations

![Customer Segmentation Summary](https://github.com/user-attachments/assets/27d93783-d1e6-4b6b-baac-e5c0b789021d)

| Segment | Marketing Opportunity |
|---|---|
| **Cluster 0 – Moderate Spenders** | Personalized online promotions and cross-selling |
| **Cluster 1 – High-Value Traditional Shoppers** | Loyalty programs, premium recommendations, and catalog campaigns |
| **Cluster 2 – Budget-Conscious Families** | Affordable bundles and value-focused promotions |
| **Cluster 3 – High-Value Multichannel Shoppers** | Personalized digital offers and targeted discount campaigns |
| **Cluster 4 – Moderate-Spending Families** | Family-oriented bundles and incentives to increase purchase frequency |

**Business Implications**

1. **Customer retention:** Prioritize Clusters 1 and 3, which demonstrate higher spending.
2. **Marketing personalization:** Tailor campaigns to preferred purchasing channels and promotional behavior.
3. **Customer activation:** Explore targeted incentives for lower-spending customers, particularly Cluster 2.
4. **Revenue growth:** Identify cross-selling opportunities among moderate-spending customers in Clusters 0 and 4.

These recommendations are hypotheses for marketing experiments rather than proven campaign outcomes.

## 6. Limitations & Future Improvements

- **Outlier removal:** IQR filtering excluded approximately 32.5% of records, potentially affecting segment representativeness.
- **Categorical encoding:** Ordinal encoding may introduce artificial relationships between categories.
- **PCA:** The selected four components retained approximately 51% of total variance.
- **Cluster validation:** Silhouette scores and cluster stability testing could strengthen the selection of K.
- **Business validation:** A/B testing could evaluate the effectiveness of segment-specific marketing strategies.

## 7. Tools & Technologies

- **Python:** Data analysis and machine learning
- **Pandas & NumPy:** Data cleaning and feature engineering
- **Matplotlib & Seaborn:** Exploratory data analysis and visualization
- **Scikit-learn:** StandardScaler, PCA, and K-Means clustering
- **Jupyter Notebook:** Analysis and documentation

## Conclusion

This project demonstrates how **data cleaning, exploratory analysis, dimensionality reduction, and unsupervised learning** can transform customer data into actionable business insights.

The five customer segments reveal meaningful differences in spending, household composition, shopping preferences, and marketing engagement, providing a foundation for more targeted and data-driven marketing strategies.
