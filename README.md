# Customer Segmentation | PCA & K-Means

**Python · Pandas · Seaborn · Scikit-learn · Customer Analytics**

## Overview

This project uses unsupervised learning to segment grocery customers by **demographics, spending, shopping channels, and campaign engagement**, with the goal of informing targeted marketing strategies.

**Dataset:** [Kaggle Marketing Campaign](https://www.kaggle.com/datasets/jackdaoud/marketing-data) · **Original:** 2,240 customers, 29 fields · **After cleaning:** 1,512 customers · **Output:** 5 segments

## Methodology

1. **Clean:** Impute 24 missing incomes with the mean; consolidate education and marital-status categories; remove four ambiguous marital-status records; filter numerical outliers using the 1.5×IQR rule.
2. **Engineer features:** Derive `Age` (2024 reference), `TotalSpent` (sum across six product categories), estimated `FamilySize`, `Is_Parent`, and number of accepted campaigns (`Total_promos`).
3. **Explore:** Examine distributions, spending patterns, purchasing channels, and correlations.
4. **Model:** Standardize features, apply **PCA (4 components)**, select **K=5** using the elbow method, and fit **K-Means** (`random_state=142`).
5. **Interpret:** Compare clusters by income, expenditure, household characteristics, purchase channels, and campaign responses.

> **Method note:** The first four PCA components explain approximately **51%**, not 70%, of the variance shown. Outlier filtering removes 728 records, which may exclude valuable high-spending customers. The notebook's 2D cluster plot also includes cluster labels in a subsequent PCA projection; this visualization should be regenerated without labels as input.

## Exploratory data analysis

### Demographics and spending

![Customer age distribution](assets/age_distribution.png)

Customers are concentrated around ages **45–60**. Three-person households are the largest group; **wine and meat** contribute the most aggregate product spending.

![Spending by household size](assets/spending_by_family_size.png)

![Feature correlations](assets/correlation_matrix.png)

**Key associations:** income and total spending (**r≈0.77**); catalog purchases and total spending (**r≈0.82**); store purchases and total spending (**r≈0.80**). These correlations are descriptive, not causal.

## Clustering results

### PCA and segment distribution

![PCA explained variance](assets/pca_explained_variance.png)

![Five clusters in projected space](assets/cluster_scatter.png)

![Customers per cluster](assets/cluster_distribution.png)

**Clusters 2 and 4 are the largest**, while Cluster 1 is the smallest.

### Five customer profiles

![Income and spending by cluster](assets/income_spending_clusters.png)

| Cluster | Profile | Purchasing behavior | Potential marketing approach |
|---|---|---|---|
| **0** | Lower-to-middle income; moderate spending; smaller households | Primarily web and store | Convenient, affordable cross-channel offers |
| **1** | High income; high spending; smaller households | Strongest catalog activity; high store activity | Premium bundles and loyalty rewards |
| **2** | Lower income; lowest spending; often three-person households | Lowest purchases across channels | Entry-level offers and low-cost reactivation tests |
| **3** | Upper-middle-to-high income; high spending; often 2–3 people | Most active web and store shoppers; frequent deal use | Omnichannel rewards and targeted promotions |
| **4** | Middle-to-upper income; moderate spending; often larger households | Web and store, less frequent than Cluster 3 | Household bundles and value-focused promotions |

### Campaign and channel behavior

![Campaign responses and deal purchases](assets/campaign_and_deals.png)

Most customers accepted **none** of the five historical campaigns. Cluster 1 contains some repeat responders; Cluster 3 shows notable deal-purchasing activity. Campaign effectiveness should be tested using **response rates**, not raw response counts alone.

![Web purchases and visits by cluster](assets/web_purchases_visits.png)

![Catalog and store purchases by cluster](assets/catalog_store_purchases.png)

**Cluster 3** has the highest average web purchase frequency, while **Cluster 1** leads catalog purchases; Clusters 1 and 3 are particularly active in stores.

### Demographic profiling

![Education composition by cluster](assets/education_clusters.png)

![Age and spending density by cluster](assets/age_spending_kde.png)

![Household size and spending density by cluster](assets/family_spending_kde.png)

Clusters **1 and 3** consist of graduate/postgraduate customers in the analysis. The density plots highlight differences in **age, household size, and spending**, but show overlapping populations rather than rigid demographic boundaries.

## Business takeaways

- **Prioritize retention** for high-spending Clusters 1 and 3, using different channels for each.
- **Test household-oriented promotions** for Clusters 2 and 4 rather than treating all customers alike.
- **Measure incremental campaign lift** through controlled experiments before scaling offers.

## Limitations and next steps

The findings are exploratory: mean imputation, ordinal encoding, estimated household size, and extensive IQR filtering can influence the resulting clusters. Future work should compare alternative preprocessing, validate cluster quality (e.g., silhouette score), select PCA dimensions using cumulative explained variance, and evaluate segment stability over time.

**Source:** [Marketing Campaign dataset](https://www.kaggle.com/datasets/jackdaoud/marketing-data)
