# Credit Risk Classification Analysis Report

**Purpose**

The purpose of the analysis was to use historical lending data from a lending services company to build a machine-learning model that can identify the credit worthiness of borrowers.

**Data**

Historical lending data was imported from `lending_data.csv`, which contained fields like `loan_size`, `interest_rate`, `borrower_income`, `debt_to_income`, `num_of_accounts`, `derogatory_marks`, `total_debt`, and `loan_status`. Healthy loans were identified as `0` and high risk loans were identified as `1` in the `loan_status` column. Objective is for the model to predict if the loan to the borrower in the testing data set would be a low or a high risk loan.

**Overview of the Analysis**

- Lending Data from the CSV file was imported to a dataframe as shown below:
  
![image](https://github.com/Anubala85/credit-risk-classification/assets/158111116/aad2b873-e850-4659-bccc-e0fb5c560ac8)

- Labels set from the `loan_status` was loaded into `y_var` while rest of the feature set was loaded into `x_var`. Used `value_counts` to check the balance of the labels and determined there were `75,036` healthy loans and `2,500` high-risk loans.
- The dataset was split into training and testing sets using the `train_test_split` module from `scikit-learn` and stored in `X_train`, `X_test`, `y_train`, and `y_test`. Assigned a random_state of 1 to ensure train/test split is consistent with same data points being applied to both the sets for various runs of the code.
- Fitted a logistic regression model using the `LogisticRegression` module from `scikit-learn` on the training dataset.
- Used the same model to make a prediction on the testing data by using the testing feature set `X_test`.

**Results**


