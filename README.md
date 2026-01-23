# Production

## Content

* [Description](#description)
* [Learning Outcomes](#learning-outcomes)
* [Contacts](#contacts)
* [Delivery of the Learning module](#delivery-of-the-learning-module)
* [Schedule](#schedule)
* [Requirements](#requirements)
* [Assessment](#assessment)

    - [Quizzes](#quizzes)
    - [Assignments](#assignments)

* [Resources](#resources)

    - [Documents](#documents)
    - [Videos](#videos)

* [Folder Structure](#folder-structure)



## Description

This module provides an overview of the Design of Machine Learning Systems embedded in data-intensive products and applications. It covers the fundamental components of the infrastructure, systems, and methods necessary to implement and maintain Machine Learning (ML) models in production. In short, we will learn techniques for building an ML model factory.

The module has two components:

* A discussion of the main issues and challenges faced in production, together with some approaches to address them.
* A live lab with demonstrations of implementation techniques.

The module covers the following areas:

* Data engineering.
* Feature engineering.
* Hyperparameter tuning.
* Model deployment.
* Model explainability.
* Logging, experiment tracking, and monitoring.

We will discuss the tools and techniques required to do the above in good order and at scale. However, we will not discuss the inner workings of models, advantages, and so on. We will also not discuss the theoretical aspects of feature engineering or hyperparameter tuning. We will focus on tools and reproducibility.

This module follows the contents of [Desinging Machine Learning Systems, by Chip Huyen](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/).

## Learning Outcomes

By the end of this module, participants will be able to:

* Describe the main components of a machine learning system.
* Explain the infrastructure required to train and test models in production.
* Implement an experiment tracking system and logging.
* Contrast and evaluate different approaches to storing and manipulating data.
* Design data flows and processes to automate the construction of ML models.


## Contacts

**Questions can be submitted to the _#cohort-8-help_ channel on Slack.**

+ **Technical Facilitator** 

    - [Jesús Calderón](https://www.linkedin.com/in/jcalderon/)
  
+ **Learning Support Team**

    - [Gayathri Girish](https://www.linkedin.com/in/gayathri-girish/)
    - [Edward Chen](https://www.linkedin.com/in/edwardchen75/)
    - [Ernani Fantinatti](https://www.linkedin.com/in/efantinatti/)
  

## Delivery of the Learning Module

This module will include live learning sessions and optional, asynchronous work periods. During live learning sessions, the Technical Facilitator will introduce and explain key concepts and demonstrate core skills. Learning is facilitated during this time. Before and after each live learning session, the instructional team will be available to answer questions about the module's core concepts. Optional work periods are to be used to seek help from peers, the Learning Support team, and to work through the homework and assignments in the learning module, with access to live help. Content is not facilitated, but rather, this time should be driven by participants. We encourage participants to come to these work periods with questions and problems to work through.
 
Participants are encouraged to engage actively during the learning module. The key to developing the core skills in each learning module is through practice. The more participants engage in coding alongside the instructional team and apply these skills in each module, the more likely they are to solidify them.

# Schedule

| Session |Date        |Topic                             |
|-----|------------|----------------------------------|
|  1  | Tue., Jan. 13, 2026    | ML System Design                 |
|  2  | Wed., Jan. 14, 2026    | Data Engineering Fundamentals    |
|  3  | Thur., Jan. 15, 2026    | Working with Training Data       |
|  --  | Fri., Jan. 16, 2026     | Work Period  |
|  --  | Sat., Jan. 17, 2026     | Work Period  |
| --  | **Sun., Jan. 18, 2026**        | **Submission deadline for Quizzes 1-3** |
| --  | **Mon., Jan. 19, 2026**        | **Submission deadline for Assignment 1** |
|  4  | Tue., Jan. 20, 2026     | Feature Engineering              |
|  5  | Wed., Jan. 21, 2026     | Model Development and Evaluation |
|  6  | Thur., Jan. 22, 2026     | Model Explanations and Monitoring|
|  --  | Fri., Jan. 23, 2026     | Work Period  |
|  --  | Sat., Jan. 24, 2026     | Work Period  |
|  --  | **Sun., Jan. 25, 2026**     | **Submission deadline for Quizzes 4-6** | 
|  --  | **Mon., Jan. 26, 2026**     | **Submission deadline for Assignment 2** | 

### Requirements

* Participants are expected to have completed Shell, Git, and Python learning modules.
* Participants are encouraged to ask questions and collaborate with others to enhance their learning experience.
* Participants must have a computer and an internet connection to participate in online activities.
* Participants must have VSCode installed with the following extensions: 
      - [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
      - [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
* Participants must [install Docker](https://docs.docker.com/engine/install/) as this module implements a Docker backend that will run a PostgreSQL server. This is intended to mimic a production-like environment. Participants may use SQLite if Docker is not an option.
* Participants must not use generative AI such as ChatGPT to generate code in order to complete assignments. It should be used as a support tool to help you find answers to questions you may have.
* We expect participants to have completed the steps in the [onboarding repo](https://github.com/UofT-DSI/onboarding/blob/main/environment_setup/README.md).
* We encourage participants to default to having their camera on at all times, and turning the camera off only as needed. This will greatly enhance the learning experience for all participants and provide real-time feedback for the instructional team. 

### Assessment

Your performance on this module will be assessed using six quizzes and two assignments. 

#### Quizzes

Quizzes will help you build key concepts in data science, data engineering, and machine learning engineering. Historically, learners take 5-10 minutes to complete each quiz, achieving an average score of +80%. 

+ Each quiz will contain material from each live learning session.
+ You will receive a link to each quiz during the respective live learning session. The links are personalized; please do not share them. If you did not receive a link, contact any member of the course delivery team.
Each quiz will contain approximately 10 questions of various types, including true/false, multiple-choice, and simple selection.
+ All quizzes are mandatory and should be submitted by their due date. 
+ The quizzes will remain open until their respective due dates, after which you will not have access to them.

#### Assignments

Assignments will help you develop coding and debugging skills. They will cover foundational skills and will extend to advanced concepts. We recommend that you attempt all assignments and submit your work, even if it is incomplete (partial submissions will earn partial marks). 

+ Each assignment should be submitted using the usual method in DSI via a Pull Request. 
+ The assignments and their respective rubrics are:

    - [Assignment 1](./02_activities/assignments/assignment_1.ipynb). [Rubric](./02_activities/assignments/assignment_1_rubric_clean.xlsx).
    - [Assignment 2](./02_activities/assignments/assignment_2.ipynb). [Rubric](./02_activities/assignments/assignment_2_rubric_clean.xlsx).


#### Grades

All participants will receive a pass or fail mark. For this course, a score of 60 points is required to receive a "pass" mark. The score will be determined as follows:

+ Quizzes' average score - 60%
+ Assignment 1 - 20%
+ Assignment 2 - 20%

Assignments' assessment can be transformed into a numeric grade using:

+ Complete - 100 points
+ Incomplete / Partially Complete - 50 points
+ Missing / Not submitted - 0 points

For example, a learner with the following grades would receive "pass":

+ Quizzes 80
+ Assignment 1 - Complete (100)
+ Assignment 2 - Incomplete (50)
+ (0.6 * 80) + (0.2 * 100) + (0.2 * 50) = 48 + 20 + 10 = 78 > 60

A different learner with grades as shown below would receive "fail":

+ Quizzes 80
+ Assignment 1 - Incomplete (50)
+ Assignment 2 - Missing (0)
+ (0.6 * 80) + (0.2 * 50) + 0 = 48 + 10 + 0 = 58 < 60

## Resources



### ML Models

Books that mainly discuss learning methods, their applications, and limitations.

- Burkov. [The Hundred-Page Machine Learning Book](https://themlbook.com/). A practical and concise overview of ML methods. It has good coverage of the learning process and a good focus on the classical algorithms. It makes a good desktop reference.
- Goodfellow, Bengio, and Courville. [Deep Learning](https://www.deeplearningbook.org/). An in-depth discussion of deep learning methods. 
- James, Witten, Hastie, Tibishirani, and Taylor. [An Introduction to Statistical Learning with Applications in Python](https://www.statlearning.com/). This is the introductory and updated version of Hastie and Tibishirani's classic book. It contains deep discussions, extensive examples, and formal arguments.
- Witten, Frank, Hall, Pal, and Foulds. [Data Mining: Practical Machine Learning Tools and Techniques](https://ml.cms.waikato.ac.nz/weka/book.html). A great introductory textbook, written from a practical perspective. The book has been updated over the year and it covers a wide spectrum of models (not only Neural Nets/Deep Learners).

### ML Engineering

Books that discuss how to put learning methods in production, including training, deployment, monitoring. As well as more architecture-oriented references.

- Burkov. [Machine Learning Engineering](https://www.mlebook.com/wiki/doku.php). Similar to Burkov's book above, but for ML Engineering.
- Khun and Silge. [Tidy Modelling with R](https://www.tmwr.org/). The book implements its examples in R, but the discussion about models, evaluation techniques, and pipelines is highly worthwhile. Both, Julia Silge and Max Khun, are brilliant data scientists and great communicators. 
- Kleppmann. [Designing Data-Intensive Applications](https://dataintensive.net/). A great in-depth resource about the techniques for building data-intensive applications.


### Advanced Topics

References for specific topics like Feature Engineering, Conformal Prediction, and Model Interpretability.

- Khun and Johnson. [Feature Engineering and Selection: A Practical Approach for Predictive Models](http://www.feat.engineering/). Discusses feature engineering methods, their evaluation, and ideas on how to implement them. This one does not contain code, only ideas. 
- Manokhin. [Practical Guide to Applied Conformal Prediction in Python](https://www.packtpub.com/en-us/product/practical-guide-to-applied-conformal-prediction-in-python-9781805122760). Discusses conformal prediction methods with Python, which will allow you to clearly define uncertainty of the predictions that you obtain from ML methods.
- Molnar. [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/). An overview of model-agnostic explainability/interpretability methods.


### Online Resources and Repositories

- [Chip Huyen's DMLS repo on GitHub](https://github.com/chiphuyen/dmls-book)
- [Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Dask Documentation](https://docs.dask.org/en/stable/)
- [ML Flow Documentation](https://mlflow.org/docs/latest/)
- [Docker Installation](https://docs.docker.com/engine/install/)
- [Docker Cheatsheet](https://docs.docker.com/get-started/docker_cheatsheet.pdf)

### Videos

- [What is Docker?](https://www.youtube.com/watch?v=Gjnup-PuquQ)
- [Docker Playlist](https://www.youtube.com/playlist?list=PLe4mIUXfbIqaYmsoFahYCbijPU4rjM38h)

## Folder Structure

```markdown
.
├── .github
├── 01_materials
├── 02_activities
├── 03_instructional_team
├── 04_this_cohort
├── 05_src
├── .gitignore
├── LICENSE
├── SETUP.md
├── pyproject.toml
└── README.md
```

* **.github**: Contains issue templates and pull request templates for the repository.
* **materials**: Module slides and interactive notebooks (.ipynb files) used during learning sessions.
* **activities**: Contains graded assignments, exercises, and homework to practice concepts covered in the learning module.
* **instructional_team**: Resources for the instructional team.
* **this_cohort**: Additional materials and resources for cohort three.
* **src**: Source code, databases, logs, and required dependencies (requirements.txt) needed during the module.
* **.gitignore**: Files to exclude from this folder, specified by the Technical Facilitator
* **LICENSE**: The license for this repository.
* **SETUP.md**: Contains the steps required to set up this repo for the module.
* **pyproject.toml**: Tells Python which packages this repo needs to run.  
* **README.md**: This file.