
# Production

## Contents:

1. Description
2. Learning Outcomes
3. Logistics
4. Marking Scheme
5. Policies
6. Folder Structure
7. Acknowledgements and Contributions


## Achnowledgement

We wish to acknowledge this land on which the University of Toronto operates. For thousands of years it has been the traditional land of the Huron-Wendat, the Seneca, and most recently, the Mississaugas of the Credit River. Today, this meeting place is still the home to many Indigenous people from across Turtle Island and we are grateful to have the opportunity to work on this land.


## Description

The course was created by the University of Toronto's Data Science Institute. The course provides an overview of the Design of Machine Learning Systems which are embedded within data-intensive products and applications. It covers the fundamental components of the infrastructure, systems, and methods necessary to implement and maintain Machine Learning (ML) models in production. In short, we will learn methods to build a factory of ML models. 

The course has two components: 

+ A discussion of the main issues and challenges faced in production, together with some approaches to address them.
+ A live lab with demonstrations of implementation techniques. 

The course covers the following areas:

+ Data engineering.
+ Feature engineering.
+ Hyperparameter tuning.
+ Model deployment.
+ Model explainability.
+ Logging, experiment tracking, and monitoring.

We will discuss the tools and techniques required to do the above in good order and at scale. However, we will not discuss the inner working of models, advantages, and so on. As well, we will not discuss the theoretical aspects of feature engineering or hyperparameter tuning. We will focus on tools and reproducibility.

## Learning Outcomes

By the end of this course, a student will be able to:

+ Describe the main components of a machine learning system.
+ Explain the infrastructure required to train and test models in production.
+ Implement an experiment tracking system and logging.
+ Contrast and evaluate different approaches of storing and manipulating data.
+ Design data flows and processes to automate the construction of ML models.


## Logistics

### Course Contacts

+ Instructor: Jesús Calderón (he/him)

  * dsi.production.course [at] gmail.com
    
    - This email is exclusively for the course. 
    - I will monitor this email and respond within 24 hours.

  * [LinkedIn](https://www.linkedin.com/in/jcalderon/) 

+ TA: TBD

### Delivery instructions

+ The workshop will be held over three weeks on the dates outlined below. 
+ Most days, we will review slides for about one hour, take a short break and continue with the technical discussion.
+ There are Jupyter notebooks in the repo to follow along in the coding sessions. 
+ We encourage you to participate and ask questions.

### Technology Requirements

1. A standard PC with Python installed. Ideally, an account with admin rights to this PC. 
2. The examples are not computationally intensive and can be further reduced if performance is an issue.
3. The course is implemented with a Docker backend that will run a PosgreSQL server. This is intended to mimic a production-like environment. Use SQLite if Docker is not an option.
4. Camera is optional although highly encouraged. We understand that not everyone may have the space at home to have the camera on.


# Course Schedule

## Part 1:	Introduction

### 1.1 Overview of ML Systems 
	
+ When to Use ML
+ ML in Production
+ ML vs Traditional Software
	
### 1.2 Introduction to ML System Design 
	
+ Business and ML Objectives
+ Requirements of Data-Driven Products
+ Iterative Process
+ Framing ML Problems 
	
### 1.3 Project Setup

+ Git, authorization, and production pipelines.
+ VS Code and Git.
+ Python virtual environments.
+ Repo File Structure.
+ Branching Strategies.
+ Commit Messages.



## Part 2:	Data Engineering Fundamentals


### 2.1. Fundamentals of Data Engineering
	
+ Data Sources
+ Data Formats  
+ Data Models 
+ Data Storage and Processing
+ Modes of Data Flow

### 2.2 An Initial Data Flow

+ Jupyter notebooks and source code. 
+ Logging and using a standard logger.
+ Environment variables.
+ Getting the data.
+ Schemas and index in dask.
+ Reading and writing parquet files.
+ Dask vs pandas: a small example of big vs small data.



## Part 3:	Working with Training Data

### 3.1. Training Data 
	
+ Sampling
+ Labeling
+ Class Imbalance
+ Data Augmentation

	
### 3.2 A Training Pipeline

+ Sampling in Python.
+ An initial training pipeline.
+ Modularizing the training pipeline.
+ Decoupling settings, parameters, data, code, and results.

### Part 4:	Feature Engineering

#### 4.1. Feature Engineering 

+ Common Operations
+ Data Leakage
+ Feature Importance 
+ Feature Generalization
	

#### 4.2 Feature Pipelines

+ Transformation Pipelines
+ Encapsulation and parametrization


## Part 5:	Model Development and Evaluation

### 5.1. Model Development and Offline Evaluation

+ Model Development and Training 
+ Ensembles 
+ Experiment Tracking and Versioning 
+ Model Offline Evaluation 

### 5.2 Experiment Tracking

+ Experiment tracking
+ Hyperparameter Tuning


### Part 6:	Deployment and Model Explanations

#### 6.1. Model Deployment

+ ML Deployment Myths
+ Batch Prediction vs Online Prediction

#### 6.2 Explainability Methods

+ Partial Dependence Plots
+ Permutation Importance
+ SHAP Values


## Part 7:	Data Distribution Shifts

### 7.1. Data Distribution Shifts and Monitoring 

+ ML System Failures
+ Data Distribution Shifts
+ Monitoring and Observability
	
### 7.2 Testing Distributions Shifts

+ Python implementation

### Part 8:	Infrastructure and Organization

#### Topic 8.1. Infrastructure and Tooling for ML Ops
	
+ Infrastructure
+ ML Ops

#### Topic 8.2. The Human Side of ML

+ Roles and skills
+ Organization


# Key Dates 

## Class schedule:

Week 15 

+ Monday, Mar 11, 6 pm - 8:30 pm
+ Tuesday, Mar 12, 6 pm - 8:30 pm 
+ Wednesday, Mar 13, 6 pm - 8:30 pm
+ Thursday, Mar 14, 6 pm - 8:30 pm 

Week 16

+ Tuesday, Mar 19, 6 pm - 8:30 pm 
+ Wednesday, Mar 20, 6 pm - 8:30 pm 
+ Thursday, Mar 21, 6 pm - 8:30 pm
+ Saturday, Mar 23, 9 am - 11:30 am

## Assignments

+ Assignment 1 due on March 15.
+ Assignment 2 due on March 19.
+ Assignment 3 due on March 23.


# Policies

+ Evaluation.

  - Quizzes will follow every session. They include multiple choice, multiple selection, and true/false questions related to the day's quesitons. The quizzes are not only assessment, but an integral part of learning. I recommend that you do not leave them all to the very last minute.
  - There will be three coding assignments. 

+ Reading.
+ Attendance. 

+ The course has a live-coding component. 
+ Students are expected to follow along with the coding, creating files and folders to navigate and manipulate. Students should be active participants while coding and are encouraged to ask questions throughout.


+ [U of T Holiday Schedule](https://people.utoronto.ca/news/holiday-schedule-2022-23-and-2023-24/)


# Folder Structure

Below are the folders contained in this repo with a description of what they contain and information on how to use them.

+ `assignments`: assignment files.
+ `config`: configuration files for experiments.
+ `docs`: all notes and quizzes.
+ `notebooks`: Jupyter notebooks
+ `src`: code.

# Contributions 

* We welcomes issues, enhancement requests, and other contributions. To submit an issue, use [GitHub
issues](https://github.com/UofT-DSI/production/issues).
