---
title:  Model Deployment
subtitle: Production
author: Jesús Calderón
---


# Introduction

## Agenda

::::::{.columns}
:::{.column}

**6.1 Model Deployment and Prediction Service **
	
+ ML Deployment Myths and Anti-Patterns
+ Batch Prediction vs Online Prediction
+ Model Compression
+ ML in the Cloud

:::
:::{.column}

**6.2 Deployment**

+ Deployment to a DB.
+ Deployment to Pickle file.
+ Power BI and Jupyter.

:::
::::::


## About These Notes

::::::{.columns}
:::{.column}

These notes are based on Chapters 7 of [*Designing Machine Learning Systems*](https://huyenchip.com/books/), by [Chip Huyen](https://huyenchip.com/).

:::
:::{.column}

![](../img/book_cover.png)

:::
::::::


## Reference Architecture

![Aggrawal et al. (2020)](../img/flock_ref_arhitecture.png)

## Deployment

::::::{.columns}
:::{.column}

+ Deploying a model is to make it useable by allowing users to interact with it through an app or by using its results for a purpose in a data product (BI visuals, reports, data views).
+ Deployment is a transition of development to production environment. 


:::
:::{.column}

+ There is a wide range of production environments: from BI to live applications serving millions of users.
+ Engaging with users in formal or informal feedback conversations is useful, although not always possible.


:::
::::::


## Deployment Myths and Anti-Patterns

::::::{.columns}
:::{.column}

**1. You only deploy one or two ML models at a time**

+ Infrastructure should support many models, not only a few.
+ Many models can interact and we also need a way of mapping these interactions.
+ Ride sharing app: 

    - 10 models: ride demand, driver availability, estimated time of arrival dynamic pricing, fraud, churn, etc.
    - 20 countries.


:::
:::{.column}

**2. If we don't do anything, mode performance stays the same**

+ Software does not age like fine wine.
+ Data distribution shifts: when the data distribution that the trained model is different from the one during training.

:::
::::::


## Deployment Myths and Anti-Patterns (cont.)

::::::{.columns}
:::{.column}

**3. You won't need to update your models as much**

+ Model performance decays over time.
+ Deploy should be easy:

    - Development environment should resemble as close as possible the production environment.
    - Infrastructure should be easier to rebuild than to repair.
    - Small incremental and frequent changes.


:::
:::{.column}

**4. Most ML engineers don't need to worry about scale**

+ Scale means different things to different applications.
+ Number of users, availability, speed or volume of data.

:::
::::::

## Batch Prediction Vs Online Prediction

::::::{.columns}
:::{.column}


**Online Prediction**

- Predictions are generated and returned as soon as requests for these predictions arrive.
- Also known as on-demand prediction.
- Traditionally, requests are made to a prediction service via a RESTful API.
- When requests are made via HTTP, online prediction is known as *synchronous prediction*. 


:::
:::{.column}

**Batch Prediction**

- Predictions are generated preiodically or whenever triggered.
- Predictions are stored in SQL tables or in memory. They are later retrieved as needed.
- Batch prediction is also known as asynchronous prediction.

:::
::::::


## Model Prediction Service

::::::{.columns}
:::{.column}

Three types of model prediction or inference service:

+ Batch prediction: uses only batch features.
+ Online prediction that uses only batch features (e.g., precomputed embeddings).
+ Online streaming prediction: uses batch features and streaming features.

:::
:::{.column}

![Batch Prediction (based on Huyen 2021)](./img/batch_prediction.png)

:::
::::::


## Model Prediction Service (cont.)

::::::{.columns}
:::{.column}

![Online Prediction (based on Huyen 2021)](./img/online_prediction.png)
:::
:::{.column}

![Streaming Prediction (based on Huyen 2021)](./img/streaming_prediction.png)

:::
::::::
