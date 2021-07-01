# CREDIT CARD CUSTOMER SEGMENTATION USING CLUSTERING ALGORITHM

Scikit-Learn Team Members :
1. Lidya Kurniawati Tjahja - [Email](lidyaktjahja@gmail.com) | [LinkedIn](https://www.linkedin.com/in/lidyaktjahja/) | [GitHub](https://github.com/lidyaktjahja)
2. Stefanus Ivan Laksono - [Email](stefanusivanlaksono@gmail.com) | [LinkedIn](https://www.linkedin.com/in/stefanusivanlaksono/) | [GitHub](https://github.com/stefanusivanlaksono)

Source : <a href="https://www.kaggle.com/arjunbhasin2013/ccdata">Credit Card Dataset for Clustering</a>

---

# Background 
In this project, we position ourselves as part of the Data Scientist team at PWDK Bank. We were assigned to the marketing division to segment credit card users based on credit card usage in the last 6 months. We will help the marketing division to know how many types they should make and what is our customers behavior using Machine Learning (ML) solutions.

# I. Problem Identification
## 1. Problem Definition

Credit card business competition is very tight. Customers can easily switch to another credit card that has a lower overall fees. According to <a href="https://www.dbmarketing.com/articles/Art175.htm">How to Retain a Credit Card Customer</a>, it costs about 80 dollars to get a new credit card customer who would returns about 120 dollars a year in profit to the company only if they keep the card. If he drops the card after a few weeks or doesn't use the card, the company will lose the customer acquisition cost (CAC) plus some more money when trying to reactivate them.

This shows that the cost of acquiring new customers is higher than retaining the existing ones. The right retention strategy can increase the company's chances of retaining its customers and further reduce the estimated loss that will be borne by the company. Customer loyalty is a key factor to survive in this business competition. We can increase the loyalty of our customers by understanding their needs first.

Currently company only have one type of credit card. In order to serve customers better, company plan to release new type of credit card based on customer's needs. But, we still didn't know how many credit card we must issues or what kind of benefits we must give to which customers.

The expected output of this project is a customer cluster based on current data and situation using ML. However, due to our time, budget, and data constraints, we limit the ability of our model in this project to predict outputs for clusters only, because more detailed information requires more data.
  
## 2. Business Objectives

The business objectives that we want to achieve through this project are as follows:
- Know how our customer's behavior
- Create one or few cluster of customer type based on their credit card usage.
- Develop one or few new credit card products and details based on customer segmentation
- To help the marketing team to check if any customers might churn to help them to follow up.

The ML system to be built must be able to support this goal by providing more specific customer clusters based on their behavior.

## 3. Data Requirements

The target we set is customer segmentation based on their transaction behavior. The information needed to create a cluster is their transaction using a credit card in the past few months. For example, their credit limit, balance, purchases, one off purchases, installment purchases, payments, minimum payments, etc.

## 4. Analytic Approach
### Machine Learning Techniques
Since we don't know the cluster or the label yet then this problem can be addressed as Unsupervised Learning. Unsupervised learning here to be more specific is about clustering. We will enter the data into the program, then based on some algorithm the program will try to group it which will then return some data clusters. 

### Risk
There are two possible risks that may be caused by incorrect predictions of the ML model:
- The first scenario is when we group customers to a lower level when they should be in a higher cluster level. This can cause the Bank to lose the opportunity to make a profit. i.e. if customers are at a higher level, they tend to spend more money which leads to company profits. Or, it can lead to customer churn if they see a competitor offering a better profit.
- The second scenario is when customers are grouped at a higher level when they should be at a lower level. This can cause the Bank to suffer losses if the customer cannot pay his credit card. Or it can lead the company to opportunity loss if the customers didn't use the money.

### Performance Measure
Performance measures for evaluating the ML model are WCSS (Within Cluster Sum of Squares) and silhouette scores.

## 5. Action ?????????
The business user can utilize the prediction result by comparing it with the appraisal value given by the AMC to determine a reasonable property value.

## 6. Value  ??????????

The values created from the project are the improvement in the credit card service to give benefit to customers based on transaction.

---

# II. Data Understanding

Started with importing the dataset which is credit_card.csv. Then, we continued to handle missing datapoints, understand each features, and choose relevant features to be used in the modelling phase. From this dataset we found some missing value in MINIMUM_PAYMENTS and CREDIT_LIMIT. 
CUST_ID | BALANCE | BALANCE_FREQUENCY | PURCHASES | ONEOFF_PURCHASES | INSTALLMENTS_PURCHASES | CASH_ADVANCE | PURCHASES_FREQUENCY | ONEOFF_PURCHASES_FREQUENCY | PURCHASES_INSTALLMENTS_FREQUENCY | CASH_ADVANCE_FREQUENCY | CASH_ADVANCE_TRX | PURCHASES_TRX | CREDIT_LIMIT | PAYMENTS | MINIMUM_PAYMENTS | PRC_FULL_PAYMENT | TENURE
-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----
C10001 | 40.900749 | 0.818182 | 95.4 | 0 | 95.4 | 0 | 0.166667 | 0 | 0.083333 | 0 | 0 | 2 | 1000 | 201.802084 | 139.509787 | 0 | 12
C10002 | 3202.467416 | 0.909091 | 0 | 0 | 0 | 6442.945483 | 0 | 0 | 0 | 0.25 | 4 | 0 | 7000 | 4103.032597 | 1072.340217 | 0.222222 | 12
C10003 | 2495.148862 | 1 | 773.17 | 773.17 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 12 | 7500 | 622.066742 | 627.284787 | 0 | 12
C10004 | 1666.670542 | 0.636364 | 1499 | 1499 | 0 | 205.788017 | 0.083333 | 0.083333 | 0 | 0.083333 | 1 | 1 | 7500 | 0 |  | 0 | 12
C10005 | 817.714335 | 1 | 16 | 16 | 0 | 0 | 0.083333 | 0.083333 | 0 | 0 | 0 | 1 | 1200 | 678.334763 | 244.791237 | 0 | 12

Below is the definition of each features:
- `CUST_ID` - Identification of credit card holder
- `BALANCE` - Balance amount left in their account to make purchases
- `BALANCE_FREQUENCY` - How frequently the balance is updated, score between 0 and 1 
  - 1 = frequently updated
  - 0 = not frequently updated
- `PURCHASES` - Amount of purchases made from account 
- `ONEOFF_PURCHASES` - Maximum purchase amount done in one-go
- `INSTALLMENTS_PURCHASES` - Amount of purchase done in installment
- `CASH_ADVANCE` - Amount of cash money user take from credit card
- `PURCHASES_FREQUENCY` - How frequent the purchases are being made, score between 0 and 1
  - 1 = frequently updated
  - 0 = not frequently updated
- `ONEOFF_PURCHASES_FREQUENCY` - How frequent purchases are happening in one-go
  - 1 = frequently updated
  - 0 = not frequently updated 
- `PURCHASES_INSTALLMENTS_FREQUENCY` - How frequent purchases in installments are being done
  - 1 = frequently updated
  - 0 = not frequently updated 
- `CASH_ADVANCE_FREQUENCY` - How frequent user take money from credit card
- `CASH_ADVANCE_TRX` - Number of transactions made with "Cash in Advanced" 
- `PURCHASES_TRX ` - Number of purchase transactions made 
- `CREDIT_LIMIT` - Limit of credit card for user
- `PAYMENTS` - Amount of payment done by user
- `MINIMUM_PAYMENTS` - Minimum amount of payments made by user
- `PRC_FULL_PAYMENT` - Percent of full payment paid by user
- `TENURE` - Tenure of credit card service for user

Handling missing value, for missing value in credit limit is dropped because there is only 1 missing value in this data. Missing value in MINIMUM_PAYMENTS we put condition to impute the missing value. Some of the data has zero payment, so we assumed their minimum payments is also zero. For customers whose PAYMENTS is not zero we put minimum payments to be the same as PAYMENTS. 
Reviewing the data, there are also few column that is not needed for modeling. So, we drop CUST_ID and TENURE.
After handling missing value, impute the missing values, and remove unecessary columns, this process results in a new clean dataset that would be used for this project.

---
  
# III. Data Preparation

At feature engineering or feature selection, we focused on selecting relevant features by checking the importance of each features to each clusters.
In the feature selection process the first step we took was try to train our model with few algorithm, in this case we use K-MEANS, Agglomerative Hiearchical Clustering (AHC), Gaussian Method Model (GMM).
We try to train the data using the model above to get clusters. Then we try to train the new labeled data to get the feature importances. We split the dataset into training and test set. 
We use Decision Tree and Random Forest Classifier to get the feature importances for modeling. 

---
  
# IV. Modeling

In the modeling process we tried several base models to clusters. After we obtained the best model for this project, we did residual analysis to evaluate the model performance. 

For the modeling process we tried these base  to obtain clusters : 
  - K-Means
  - Agglomerative Hierarchical Clustering
  - Gaussian Method Model
  
---
  
# V. Evaluation
Masukin summary/ recommendation & workflow
We use Agglomerative Hierarchical Clustering Model as our final model as it gives the better cluster result.

<p float="center ">
<img src="https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/blob/main/Model/Model.png" width="500" />
  
Please refer to: <a href='https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/blob/main/Jupyter%20Notebook/Modeling_Final%20with%20CatBoost%20(without%20Onehot).ipynb'>Modeling_Final with CatBoost (without Onehot).ipynb</a>

### REFFERENCE 
https://www.dbmarketing.com/articles/Art175.htm



    
