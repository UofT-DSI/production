---
title:  Training
subtitle: Production
author: Jesús Calderón
---

# Introduction

## Agenda

**3.1 Working with Training Data**
	
+ Sampling
+ Labeling
+ Class Imbalance
+ Data Augmentation



**Assignment: Setting Up a Code Repository**

+ Implement different types of sampling
+ Some recoding exercise using sci-kit learn
+ Conformal prediction?

# Sampling

## Why Sample?

::::::{.columns}
:::{.column}

+ Sampling is embedded across the ML lifecycle: data exploration, train/validation/test split, etc.
+ Sometimes, sampling is necessary:

    - We do not have access to all possible data in the real world.
    - It is unfeasible, costly, or otherwise impractical to use all data.
    - Accomplish a task faster and cheaper: experiment with a new model, explore data, etc.

:::
:::{.column}

There are two families of sampling:

+ Nonprobability sampling.
+ Random sampling.

:::
::::::

## Nonprobability Sampling

::::::{.columns}
:::{.column}

+ Generally, it is a bad idea to select data to train ML methods using this family of sampling methods, but some of them are popular.

+ Convenience sampling

    - Select data based on their availability.
    - Popular and conveninet: fast, inexpensive, practical.
    - Not scientific and does not offer guarantees.

+ Snowball sampling

    - Future samples are selected based on existing samples. 
    - Sampling in social media (or other) networks: select a base sample of accounts, then expand the sample by adding the accounts they follow, and so on.

:::
:::{.column}

+ Judgement sampling

    - Experts decide what samples to include.
    - A.K.A.: risk-based, SME, subjective, etc.

+ Quota sampling

    - Select samples based on predefined and heuristic quotas.
    - Example: select 100 responses from all age groups, without considering the proportional representation of age groups.

:::
::::::

## Random Sampling

::::::{.columns}
:::{.column}

+ Simple Random Sampling

    - All potential samples in the population have equal probabilities of being selected. 
    - Advantage: Easy to implement.
    - Disadvantage: Rare or infrequent categories of data may not appear in the selection: if a class appears in 0.01% of the data and we randomly select 1% of the population, we may not get representation of this minority class.

+ Stratified Sampling

    - First, divide the population into groups we care about, then sample from each group separately.
    - Each group is called a *stratum* and this method is called *stratified sampling*.
    - Advantage: the distribution of groups in the population is reflected in the sample.
    - Particularly important for selecting training, validation, and test sets.
    - This method is not always possible (multilabel cases, for example).

:::
:::{.column}

+ Weighted Sampling

    - Each sample is given a weight, which determines the probability of it being selected.
    - This method allows us to leverage domain exertise.
    - Can be used to adjust samples that are coming from a different distribution than the original data:

        - Assume the data contains 25% red samples and 75% blue samples. 
        - We know that the actual distribution is closer to 50% red and 50% blue.
        - We can apply red weights that are three times higher than blue weights.


:::
::::::

## Reservoir Sampling


:::::::{.columns}
:::{.column}



+ Useful for streaming data where the concept of "universe" is difficult to implement.
+ Motivation: we want samples from a Tweeter feed with equal probability.
+ Objectives: 

    - Every tweet has an equal probability of being selected.
    - You can stop the algorithm at any time and the tweets are sample with the correct probability.

+ Reservoir sampling:

    - Put the first k elements into the reservoir.
    - For each incoming nth element, generate a random number i such that 1 ≤ i ≤ n.
    - If 1 ≤ i ≤ k: replace the ith element in the reservoir with the nth element. Else, do nothing.

+ Each incoming nth element has a k/n probability of being in the reservoir.

:::
:::{.column}



:::
:::::::


# Labeling

# Class Imbalance

# Data Augmentation