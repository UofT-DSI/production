# Production

## Contents:
* [Achnowledgement](#achnowledgement)
* [Description](#description)
* [Learning Outcomes](#learning-outcomes)
* [Logistics](#logistics)
  + [Course Contacts](#course-contacts)
  + [Delivery of Module](#delivery-of-module)
  + [How the Instructor will deliver](#how-the-instructor-will-deliver)
  + [Requirements](#requirements)
  + [Expectations](#expectations)
* [Course Schedule](#course-schedule)
* [Policies](#policies)
* [Resources](#resources)
  + [Documents](#documents)
  + [Videos](#videos)
  + [How to get help](#how-to-get-help)
* [Folder Structure](#folder-structure)

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

**Questions can be submitted to the #questions channel on Slack**

* Instructor: **{Name}** {Pronouns}. Emails to the instructor can be sent to {first_name.last_name}@mail.utoronto.ca.
* TA: **{Name}** {Pronouns}. Emails to the instructor can be sent to {first_name.last_name}@mail.utoronto.ca.

### Delivery of Module

+ The workshop will be held over three weeks on the dates outlined below. 
+ Most days, we will review slides for about one hour, take a short break and continue with the technical discussion.
+ There are Jupyter notebooks in the repo to follow along in the coding sessions. 
+ We encourage you to participate and ask questions.

### How the Instructor will deliver
The instructors will introduce the concepts through a collaborative live coding session usiing the Python notebooks found under `/01-slides`. All instructors will also upload any live coding files to this repository for any learners to revisit under `/live_production`.

### Requirements

* Learners are not expected to have any coding experience, we designed the learning contents for beginners.
* Learners are encouraged to ask questions, and collaborate with others to enhance learning.
* Learners must have a computer and an internet connection to participate in online activities.
* Learners must have VSCode installed with the following extensions: 
    * [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
    * [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
* Learners must install Docker as this module implements a Docker backend that will run a PosgreSQL server. This is intended to mimic a production-like environment. Use SQLite if Docker is not an option.
* Learners must not use generative AI such as ChatGPT to generate code in order to complete assignments. It should be use as a supportive tool to seek out answers to questions you may have.
* We expect learners to have completed the [onboarding repo](https://github.com/UofT-DSI/Onboarding/tree/tech-onboarding-docs).
* Camera is optional although highly encouraged. We understand that not everyone may have the space at home to have the camera on.

### Expectations
Learners are encouraged to be active participants while coding and are encouraged to ask questions throughout the module. Learners are expected to follow along with the coding, creating files and folders to navigate and manipulate. Learners should be active participants while coding and are encouraged to ask questions throughout.

# Course Schedule

## Part 1:	Introduction

Date: TBD
Quiz 1


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

Date: TBD
Quiz 2


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

Date: TBD
Quiz 3

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


## Assignment Due Dates

+ Assignment 1 due on TBD.
+ Assignment 2 due on TBD.
+ Assignment 3 due on TBD.


## Policies

* **Accessibility:** We want to provide an accessible learning environment for all. If there is something we can do to make this course more accessible to you, please let us know.
* **Course communications:** Communications take place over email or on Slack. If communicating over email, please include "DSI-Python" or similar in the subject line, e.g. "DSI-Python: pandas question"
* **Camera:** Keeping your camera on is optional.
* **Microphone:** Please keep microphones muted unless you need to speak. Please indicate your name before speaking as some Zoom configurations make it hard to tell who is talking!
* **Assessments:**
  - Quizzes will follow every session. They include multiple choice, multiple selection, and true/false questions related to the day's quesitons. The quizzes are not only assessment, but an integral part of learning. I recommend that you do not leave them all to the very last minute.
  - There will be three coding assignments. They should be submitted via email. 

## Resources
Feel free to use the following as resources:

### Documents
- [Docker Installation](https://docs.docker.com/engine/install/)
- [Cheatsheet](https://docs.docker.com/get-started/docker_cheatsheet.pdf)

### Videos
- [What is Docker?](https://www.youtube.com/watch?v=Gjnup-PuquQ)
- [Docker Playlist](https://www.youtube.com/playlist?list=PLe4mIUXfbIqaYmsoFahYCbijPU4rjM38h)

### How to get help
![image](/Steps%20to%20ask%20for%20help.png)

<hr>

## Folder Structure

```markdown
|-- 01-slides
|-- 02-assignments
|-- 03-instructors
|-- 04-data
|-- 05-src
|-- .gitignore
```

* **slides:** Course slides as interactive notebooks (.ipynb files)
* **live_production:** Notebooks from class live coding sessions
* **assignments:** Graded assignments
* **data**: Contains all data associated with the module
* **instructors:** Instructions for the Instructor on what to teach
* **src:** Contains credit experiment
* README: This file!
* .gitignore: Files to exclude from this folder, specified by the instructor
