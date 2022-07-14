---
layout: post
title: Learnings from Kaggle House Prices Competition - Part 1 
---

I wanted to try getting familiar with sklearn's machine learning pipeline, so I started applying it to the Kaggle House Prices Competition. The data comprised of housing prices in Ames, Iowa with 79 features. I ended up learning a lot through the whole process.

## Data Exploration and Cleaning

This was a great dataset (probably purposefully so, being a "Getting Started" competition) to practice a workflow for analyzing data since there were so many features. It definitely reminded me that I need to have the patience to observe all the data. I ended up going through a couple rounds of data cleaning.

To make the data cleaning workflow more streamlined in the future, I came up with this workflow:

**Steps for reviewing data**:


- Review data explanations if available!
- Review missing data
  -  *Purpose*: To omit columns that have too many missing values, identify data cleaning approach (ie., imputation or other)
    - Make sure to understand why data is missing. For example, in this dataset the data is NA if the house does not have a certain feature (ie., a fence). This is valuable information and should be cleaned to represent that, and not be tossed out.
- Visualization
  - *Purpose*: This can help understand correlations between the features and dependent variable. Additionally, this can help show any quirks of the dataset.
    - Look at distribution of output variable - is it normally distributed? If not, consider transforming the variable.
    - Make pair scatter plots of all the numeric variables.
    - Make boxplots of categorical variables with respect to the dependent variable
    - Any others that might be helpful


## The sklearn pipeline

### First, some terminology

One thing I learned is that there are many words in machine learning that sound scary but are actually very simple concepts. For example, "pipeline" sounds very scary, but it's just a fancy word to describe a way to apply a series of functions to your data, either to clean your data, transform it or model it.

Below are some of such words that I'll keep be using throughout:

| Terminology    | In "English" |
|----------------|--------------|
|Pipeline   |An object you can use to apply many functions to your data (for data cleaning, transforming or modeling)   |
|Transformer   |A function to "transform" your data, which can be as simple as scaling it   |
|Custom Transformer   |A function you write that can be applied to your pipeline   |
|Feature Engineering   |    You use the variables (features) you have to come up with new ones that may be helpful. For example if you have square footage and rooms, maybe you want to know the square footage per room.|

### The Pipeline
Using sklearn's pipeline was super streamlined! Below shows a code snippet of what the pipeline looks like. The `ColumnTransfomer()` function allows you to apply different pipelines to the data by the column name. In this case, I had 3 separate pipelines by the data type. Each pipeline is composed of sklearn functions or custom transformers.

~~~Python

# Pipeline for numeric variables
numeric_pipeline = Pipeline(steps = [
    ('postprocess', CleanRemodels()),
    ('impute', SimpleImputer(strategy = 'median')),
    ('homearea', add_home_area()),
    ('scale', MinMaxScaler())
])

# Pipeline for categorical variables
cat_pipeline = Pipeline(steps = [
    ('postprocess', clean_categorical()),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])

# Pipeline for categorical variables that should be changed into numeric
cat_ordinal_pipeline = Pipeline(steps = [
    ('postprocess', clean_categorical()),
    ('impute', SimpleImputer(strategy = 'most_frequent')),
    ('ordinal', change_to_ordinal()),
    ('scale', MinMaxScaler())
])

# Full pipeline that gets applied to data, which combines above three pipelines
# ColumnTransfomer allows different pipelines to be applied to the data by
# passing in the columns it should apply to.
full_pipeline = ColumnTransformer([
    ("cat", cat_pipeline, cat_columns),
    ("num", numeric_pipeline, num_columns),
    ("ord", cat_ordinal_pipeline, ordinal_columns)
])

housing_prepared = full_pipeline.fit_transform(training)

~~~

### Custom Transformers

A useful thing I learned is that you can create custom transfomers and add it to your pipeline. For example, I created a `add_home_area` Custom Transformer that adds the total home area to the dataset, which I decided to add for feature engineering. This is used in the numeric pipeline above.

```Python
class add_home_area(BaseEstimator, TransformerMixin):
    """
    Function to add total home area to numeric variables
    """

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        total_area = np.sum(X[:, indexes], axis = 1)

        return np.c_[X, total_area]
```
