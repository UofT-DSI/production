---
title:  Feature Engineering
subtitle: Production
author: Jesús Calderón
---


## Agenda

**4.1 Feature Engineering**
	
+ Common Operations
+ Data Leakage
+ Feature Importance 
+ Feature Generalization
	

## Agenda (cont.)

**Assignment: Setting Up a Code Repository**

+ Scikit learn pipelines
+ Time series cross validation
+ 1 scikit and 1 lime or shap


## Learned Features Versus Engineered Features

::::::{.columns}
:::{.column}

+ The promise of Deep Learning was that we no longer have to engineer features (feature or representation learning).
+ Why do we need to engineer features?

:::
:::{.column}

+ Some features can and will be automatically learned for certian use cases (vision, NLP, etc.)
+ However, the majority of ML applications are not deep learning. 

:::
::::::

## What is Feature Engineering?


+ Feature engineering is the process of choosing what information to use and how to extract this information into a format usable by ML models.
+ The purpose of feature engineering is to *transform and represent features so that their information content is best exposed* to the learning algorithm. 
+ Feature engineering can include:

  * A transformation of a feature: standardization, scale, center, log, etc.
  * An equivalent re-representation of a feature: dummy variables, one-hot-encoding, binning, etc.
  * An interaction of two or more features such as a product or ratio: for example, calculate the ratio of a loan to the value of the collateral (or its inverse), as a new feature for default prediction.
  * A functional relationship among features: Principal Components Analysis, LDA, etc. This may also include methods for imputing missing values.

# Common Feature Engineering Operations 


## Handling Missing Values

::::::{.columns}
:::{.column}

+ Missing values are a common occurrence in production data. 
+ Missing values can be of three types:

  - Missing Not At Random (MNAR): the reason a value is missing is because of the true value itself.
  - Missing At Random (MAR): the reason a value is missing is not due to the value itself, but due to another observed variable. 
  - Missing Completely At Random (MCAR): there is no pattern in missing values.

:::
:::{.column}

**Deletion**

+ The simplest way to remove missing values is deletion.
+ *Column deletion*

  - If a variable has too many missing values, remove the variable.
  - Drawback: one may remove important iformation and reduce model performance.

+ *Row deletion*

  - If a sample has missing values, then remove the sample.
  - Works when missing values are MCAR and number of missing values is small.
  - Drawbacks: 
      
      * Will not work when MNAR data is present. 
      * Removing rows can create biases.

:::
::::::


## Imputation 

::::::{.columns}
:::{.column}



+ Impute missing values using default values: missing strings, filled with "".
+ Use a statistic like  mean, media, or mode: fill missing temperature with the mean temperature for the time of day within a certain window.
+ Domain specific: if prices are liquid, use last available price.
  
:::
:::{.column}

+ Model-based: if two variables are correlated and one of them have missing values, model the relationship and use model results for imputation. 
+ Flag imputed missing values.
+ In general, avoid filling missing values with possible (fixed) values. Example: missing number of children should not be filled with 0, a possible value.

:::
::::::

## Scaling


::::::{.columns}
:::{.column}

+ Objective is to obtain values of similar magnitude.
+ Scaling makes variables a "standard size". It benefits algorithms that are scale sensitive and generally does not hurt algorithms that are scale insensitive. 
+ There is little downside to scaling features, in general.
+ Min-Max scaling to obtain values in the range [0, 1]:

$$
x' = \frac{x- min(x)}{max(x)-min(x)}
$$


:::
:::{.columns}

+ Scaling to an arbitrary range [a, b]:

$$
x' = a+\frac{(x- min(x))(b-a)}{max(x)-min(x)}
$$

+ If we believe that the variable is normally distributed, it may be helpful to use:

$$
x' = frac{(x- mean(x))}{std(x)}
$$


+ Warning: scaling is a common source of data leakage.
+ Scaling requires global statistics that may be expensive to calculate.

:::
::::::

## Discretization

::::::{.columns}
:::{.column}

+ 

:::
:::{.column}


:::
:::::::



## Recoding Variables

::::::{.columns}
:::{.column}

+ From the perspective of variables and values, we sometimes talk about *code* or, more frequently, how a variable is *encoded*. 
+ The *encoding of a variable* is the representation of the data. 
+ Some of the models that we use will often benefit or will require that data be *encoded* in a form that is better suited for the model's inputs. 
+ Encoding (or recoding) many times involves changing a variable type: creating dummy variables implies converting information in a categorical variable to a numeric variable, for example.

:::
:::{.column}


**Discretization**

+ Discretization, quantization, or binning is the process of turning a continuous feature into a discrete feature. 
+ Discretization is performed by creating buckets for the given values.
+ A disadvantage of discretization is that cateogrization introduces discontinuities at the category boundaries. 

:::
::::::

## Encoding Categorical Features

::::::{.columns}
:::{.column}

+ Categories are not static: categories change over time.

  - Categories were not represented in training data.
  - New categories may appear over time.
  
+ It is generally a good idea to consider the category UNKNOWN.

:::
:::{.column}

+ In some cases, UNKNOWN labels may refer to samples that do not belong together: two new brands may not target the same market, new products, new IP addresses, new user accounts, etc.
+ One solution is the hashing trick:

  - Use a hash function to generate a has for every category.
  - The hashed value will become the index of the category.
  - Some collisions may occur, but the overloading of UNKNOWN cateogry is reduced.


:::
::::::


## Feature Crossing

::::::{.columns}
:::{.column}

+ Feature crossing is a technique to combine two or more features to generate new features.
+ This may be 
+ We may also benefit from establishing interaction terms. This type of transformation is primarily intended for numeric data but may be applied to categorical data after being transformed into dummy variables.

:::
:::{.column}


:::
::::::

# Interactions



Interaction variables typically are used to capture the joint contribution to our predictive model of two or more variables after accounting for their individual contributions. A majority of cases will result in the model benefiting only marginally from these terms; however, they are fundamental in some contexts: for example, loan value and collateral value are typically included in default prediction models, together with their interaction term. 

You can specify interaction variables using `step_interact()` and denoting the interaction with `:`, for instance `step_interact(~loan:collateral)`. 

## Dummy Variables

Dummy variables are a form of encoding categorical data. For example, if we had a data set that includes a variable "day of the week" with 5 values, Monday through Friday, we can re-code it as a dummy as follows:

```{r dummy}
workday <- tibble(day_of_week = c('Mon', 
                                  'Tue',
                                  'Wed',
                                  'Thu',
                                  'Fri'))


dummy_rec <-
  recipe(~ day_of_week , data = workday) %>%
  step_dummy(day_of_week) 

dummy_rec_prep <- prep(dummy_rec) %>%
  bake(new_data = NULL)

dummy_rec_prep %>%
  kable()
```

There are several things to notice from the result above:

+ Dummy variables are binary: they will take a value of 0 or 1.
+ Dummy variables are numerical and not categorical, factor, boolean, etc.
+ If the original variable contained C levels, then we will get C-1 levels by default. For instance, our example had five levels (one per weekday), but the resulting dummy representation only has four. The reason is that we can back out the fifth value since we know that when all four values are 0, then the fifth value should be 1. This avoids an undesirable situation for some methods called colinearity (one variable can be obtained as a linear function of others). Colinearity is one form of observing information redundancy.
+ When recoding into dummy variables, the first value is dropped.
+ If the original value is missing, then all dummy variables are `NA`.
+ If the data contains a novel value (a value that it hand not yet considered and encoded), then all values will be `NA`. For these cases, we can also consider `step_other()`.

## One-Hot Encoding

Some algorithms require one-hot encoding instead of dummy variables. **One-hot encoding** is similar to dummy variable, but all values receive a column. In our example, we would get five columns and not only four.

```{r one_hot}
workday <- tibble(day_of_week = c('Mon', 
                                  'Tue',
                                  'Wed',
                                  'Thu',
                                  'Fri'))


onehot_rec <-
  recipe(~ day_of_week , data = workday) %>%
  step_dummy(day_of_week, one_hot = TRUE) 

onehot_rec_prep <- prep(onehot_rec) %>%
  bake(new_data = NULL)

onehot_rec_prep %>%
  kable()
```

## Binning (Discretization)

One simple way that we can use to transform a numeric variable into a categorical variable is to place each value in a bin:

+ `step_discretize()` will create equal-weighted bins with an approximately equal number of points.
+ `step_cut()` will create bins based on provided boundary levels. 

```{r}
dt <- tibble(x = seq(1:100))

discretize_rec <- 
  recipe(dt) %>%
  step_discretize(x, num_breaks = 7)

discretize_rec_prep <- prep(discretize_rec) %>%
  bake(new_data = NULL)

discretize_rec_prep %>%
  table()
```

```{r}
dt <- tibble(x = seq(1:100))

discretize_rec <- 
  recipe(dt) %>%
  step_cut(x, breaks = c(7, 25, 70, 90))

discretize_rec_prep <- prep(discretize_rec) %>%
  bake(new_data = NULL)

discretize_rec_prep %>%
  table()
```
# Interactions

We may also benefit from establishing interaction terms. This type of transformation is primarily intended for numeric data but may be applied to categorical data after being transformed into dummy variables.

Interaction variables typically are used to capture the joint contribution to our predictive model of two or more variables after accounting for their individual contributions. A majority of cases will result in the model benefiting only marginally from these terms; however, they are fundamental in some contexts: for example, loan value and collateral value are typically included in default prediction models, together with their interaction term. 

You can specify interaction variables using `step_interact()` and denoting the interaction with `:`, for instance `step_interact(~loan:collateral)`. 

# Multivariate Transformations

Some transformations may include more complex formulations or the results of models that we use to pre-process the data. A couple of examples include:

+ `step_pca()` which extracts the principal components of a data set.
+ `step_classdist()` calculates the distance to class centroids. A *centroid* is the element-wise average of a set of data points. 


# Other Useful Functions for Feature Engineering

Other useful functions for feature engineering include:

+ If we had a case in which labels are encoded as numbers, we can use `step_num2factor()` to convert into factor.
+ `step_corr()` filters out highly correlated variables (reduces redundancyy).
+ `step_nzv()` removes variables with low variability.


As well, please refer to the [function reference for all the step functions](https://recipes.tidymodels.org/reference/index.html) available through tidymodels.