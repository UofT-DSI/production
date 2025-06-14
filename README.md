# Production

## Content
* [Description](#description)
* [Learning Outcomes](#learning-outcomes)
* [Contacts](#contacts)
* [Delivery of the Learning module](#delivery-of-the-learning-module)
* [Schedule](#schedule)
* [Requirements](#requirements)
* [Assessment](#assessment)
  + [Quizzes](#quizzes)
  + [Assignments](#assignments)
* [Resources](#resources)
  + [Documents](#documents)
  + [Videos](#videos)
* [Folder Structure](#folder-structure)
* [Acknowledgement](#acknowledgement)


## Description

This module provides an overview of the Design of Machine Learning Systems which are embedded within data-intensive products and applications. It covers the fundamental components of the infrastructure, systems, and methods necessary to implement and maintain Machine Learning (ML) models in production. In short, we will learn methods to build a factory of ML models. 

The module has two components: 

+ A discussion of the main issues and challenges faced in production, together with some approaches to address them.
+ A live lab with demonstrations of implementation techniques. 

The module covers the following areas:

+ Data engineering.
+ Feature engineering.
+ Hyperparameter tuning.
+ Model deployment.
+ Model explainability.
+ Logging, experiment tracking, and monitoring.

We will discuss the tools and techniques required to do the above in good order and at scale. However, we will not discuss the inner working of models, advantages, and so on. As well, we will not discuss the theoretical aspects of feature engineering or hyperparameter tuning. We will focus on tools and reproducibility.

## Learning Outcomes

By the end of this module, participants will be able to:

+ Describe the main components of a machine learning system.
+ Explain the infrastructure required to train and test models in production.
+ Implement an experiment tracking system and logging.
+ Contrast and evaluate different approaches of storing and manipulating data.
+ Design data flows and processes to automate the construction of ML models.


## Contacts

**Questions can be submitted to the _#cohort-3-help_ channel on Slack**

* Technical Facilitator: 
  * [Jesús Calderón](https://www.linkedin.com/in/jcalderon/)
  
* Learning Support Staff: 
  * [Dmytro Bonislavskyi](https://www.linkedin.com/in/dmytro-bonislavskyi/)
  * [Gayathri Girish](https://www.linkedin.com/in/gayathri-girish/)
  * [Ananya Jha](https://www.linkedin.com/in/jhaananya/)
  * [Shiraz Latif](https://ca.linkedin.com/in/shiraz-latif/)
  

## Delivery of the Learning Module

This module will include live learning sessions and optional, asynchronous work periods. During live learning sessions, the Technical Facilitator will introduce and explain key concepts and demonstrate core skills. Learning is facilitated during this time. Before and after each live learning session, the instructional team will be available for questions related to the core concepts of the module. Optional work periods are to be used to seek help from peers, the Learning Support team, and to work through the homework and assignments in the learning module, with access to live help. Content is not facilitated, but rather this time should be driven by participants. We encourage participants to come to these work periods with questions and problems to work through. 
 
Participants are encouraged to engage actively during the learning module. The key to developing the core skills in each learning module is through practice. The more participants engage in coding along with the instructional team, and applying the skills in each module, the more likely it is that these skills will solidify. 

# Schedule

|Live Learning Session |Date        |Topic                             |
|-----|------------|----------------------------------|
|  1  | Tue., June 3, 2025    | ML System Design                 |
|  2  | Wed., June 4, 2025    | Data Engineering Fundamentals    |
|  3  | Thur., June 5, 2025    | Working with Training Data       |
|  --  | Fri., June 6, 2025     | Work Period  |
|  --  | Sat., June 7, 2025     | Work Period  |
| --  | Sun., June 8, 2025        | Submission deadline for **Assignment 1** and **Quizzes 1-3** |
|  4  | Tue., June 10, 2025     | Feature Engineering              |
|  5  | Wed., June 11, 2025     | Model Development and Evaluation |
|  6  | Thur., June 12, 2025     | Model Explanations and Monitoring|
|  --  | Fri., June 13, 2025     | Work Period  |
|  --  | Sat., June 14, 2025     | Work Period  |
|  --  | Sun., June 15, 2025     | Submission deadline for **Assignment 2** and **Quizzes 4-6** | 

### Requirements

* Participants are expected to have completed Shell, Git, and Python learning modules.
* Participants are encouraged to ask questions, and collaborate with others to enhance their learning experience.
* Participants must have a computer and an internet connection to participate in online activities.
* Participants must have VSCode installed with the following extensions: 
    * [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
    * [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
* Participants must [install Docker](https://docs.docker.com/engine/install/) as this module implements a Docker backend that will run a PosgreSQL server. This is intended to mimic a production-like environment. Participants may use SQLite if Docker is not an option.
* Participants must not use generative AI such as ChatGPT to generate code in order to complete assignments. It should be used as a supportive tool to seek out answers to questions you may have.
* We expect participants to have completed the steps in the [onboarding repo](https://github.com/UofT-DSI/onboarding/).
* We encourage participants to default to having their camera on at all times, and turning the camera off only as needed. This will greatly enhance the learning experience for all participants and provides real-time feedback for the instructional team. 

### Assessment

Your performance on this module will be assessed using six quizzes and two assignments. 

#### Quizzes

Quizzes will help you build key concepts in data science, data engineering, and machine learning engineering. Historically, learners take 5-10 minutes to answer each quizz to obtain an average score of +80%. 

+ Each quiz will contain material from each live learning session.
+ You will receive a link to each quiz during the respective live learning session. The links are personalized, please do not share them. If you did not receive a link, contact any member of the course delivery team.
+ Each quiz will contain about 10 questions of different types: true/false, multiple choice, simple selection, etc.
+ All quizzes are mandatory and should be submitted by their due date. 
+ The quizzes will remain open until their respective due dates, after which you will not have access to them.

#### Assignments

Assignments will help you develop coding and debuging skills. They will cover foundational skills and will extend to advanced concepts. We recommend that you attempt all assignments and submit your work even if it is incomplete (partial submissions will get you partial marks). 

+ Each assigment should be submitted using the usual method in DSI via a Pull Request. 
+ The assigments and their respective rubrics are:

  - [Assignment 1](./02_activities/assignments/assignment_1.ipynb). [Rubric](./02_activities/assignments/assignment_1_rubric_clean.xlsx).
  - [Assignment 2](./02_activities/assignments/assignment_2.ipynb) . [Rubric](./02_activities/assignments/assignment_2_rubric_clean.xlsx).


#### Grades

All participants will receive a pass or fail mark. The mark will be determined as follows:

+ Quizzes' average score - 60%
+ Assginment 1 - 20%
+ Assignment 2 - 20%

Assignments' assessment can be transformed to a numeric grade using:

+ Complete - 100 points
+ Incomplete / Partially Complete - 50 points
+ Missing / Not submitted - 0 points

For this course, 60 points are required to receive a "pass" mark.

For example, a learner with the following grades would receive "pass":

+ Quizzes 80
+ Assignment 1 - Complete (100)
+ Assignment 2 - Incomplete (50)
+ (0.6 * 80) + (0.2 * 100) + (0.2 * 50) = 48 + 20 + 10 = 78 > 60

A different learner with grades as shown bellow would receive "fail":

+ Quizzes 80
+ Assignment 1 - Incomplete (50)
+ Assignment 2 - Missing (0)
+ (0.6 * 80) + (0.2 * 50) + 0 = 48 + 10 + 0 = 58 < 60

## Resources


### Documents and Repositories

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
* **README**: This file.

