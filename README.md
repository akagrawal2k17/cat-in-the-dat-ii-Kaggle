# cat-in-the-dat-ii-Kaggle
cat-in-the-dat-ii-Kaggle

**Datasets:**
  This comes from kaggle from the URL- https://www.kaggle.com/c/cat-in-the-dat-ii

**Description:**
  This datasets basically shows different types of categorical variables such as-
  1. Ordinal
  2. Nominal
  3. Binary
  4. Cyclic
  
  **Models worked on and performance comapred:**
  1. Logistic Regression with One Hot Encoding
  2. Random Forest with Lable Encoding
  3. SVD and Random Forest with One Hot Encoding
  4. Label Encoding with XGBoost
  
  **Summary:** For Nominal categorical variables we can use Lable Encoding but not with the Ordinal ones. One hot Encoding creates an sparsed array and saves memory utilizaton for big sized datasets. Tree Based algorithm support Label Encoding, but for others models we need to go with OHE.
