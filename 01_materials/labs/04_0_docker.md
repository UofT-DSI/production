# Using Docker to Set Up Experiment Tracking

+ For our work, we need an environment that closely resembles the production environment. 
+ One way to achieve this is to use containers and containerized applications. 
+ Without going into the details, you can think of a container as software that encapsulates the key features of an operating system, a programming language, and the application code.
+ Containers are meant to be portable across operating systems: a container will work the same regardless of whether the underlying Docker application is installed on a Windows, Linux, or Mac machine.
+ Containers are not Virtual Machines.
+ Docker is a popular containerization platform.

## What is Docker?

+ From product documentation:

> Docker is an open platform for developing, shipping, and running applications. Docker enables you to separate your applications from your infrastructure, allowing you to deliver software quickly. With Docker, you can manage your infrastructure in the same ways you manage your applications. By leveraging Docker's methodologies for shipping, testing, and deploying code, you can significantly reduce the time between writing code and running it in production.

## Installation

Go to [docker.com](https://www.docker.com/products/docker-desktop/). Hover over "Download Docker Desktop" and select the right version for your operating system. If you do not know how to choose, the main distinction is whether you have an ARM processor. Detailed installation instructions are below:

- [Windows users](https://docs.docker.com/desktop/setup/install/windows-install/) If you are using a regular Intel or AMD processor, use the "x86_64" or "AMD64" version.
- [Mac users](https://docs.docker.com/desktop/setup/install/mac-install/) select "Apple Silicon" or "ARM" if you have an ARM processor.
- [Linux users](https://docs.docker.com/desktop/setup/install/linux/), if you would like the Desktop UI, you can install from docker.com. Otherwise, I expect the containers to run on the Linux Docker Engine.


## General Procedure

After installation, to set up services using containers, we will do the following:

1. Download an image from [Docker Hub](https://hub.docker.com/) or an equivalent image repository.
2. If required, set up a volume to [persist data](https://docs.docker.com/guides/walkthroughs/persist-data/).
3. Redirect ports as needed.
4. Start the container.

In our course, we will set up the following services:

+ MLFlow: an experiment tracking system. MLFlow requires two backends: a database and an object store.
+ PostgreSQL: a database management system.
+ MinIO: an object store that resembles S3 buckets in AWS.

## Starting the Containers

+ To run the process above, first navigate to the `./05_src/experiment_tracking/` folder.
+ The first time that you set up the containers, you will need to build the MLFlow image. You can build the required image with `docker compose build`. 
+ After building a local image for MLFlow, run `docker compose up -d`. 
+ The flag `-d` indicates that we will do a headless run. 
+ Notice that the containers are set to always restart. You can remove the option or turn the containers off manually. Be aware that if you leave this option on, the containers will run whenever Docker Desktop restarts.

## Stopping the Containers

+ To stop the containers use (from `./05_src/experiment_tracking/`): `docker compose stop`.
+ Alternatively, you can bring all images down, including their volumes, with: `docker compose down -v`. 

    - The `-v` flag removes volumes. 
    - It is the best option when you do not need the data any more because **it will delete the data in your DB **. 


## Connecting to the MLFlow UI

+ MLFlow provides a convenient interface accessible at [http://localhost:5001](http://localhost:5001).

<div><img src="./images/01_mlflow.png" height=450></div>


## Connecting to PgAdmin

+ PgAdmin4 is management software for PostgreSQL Server.
+ You can open the local implementation by navigating to [http://localhost:5051](http://localhost:5051/). You will find a screen like the one below.

<div><img src="./images/01_pgadmin_login.png" height=450></div>

+ Login using the credentials specified in the file `./05_src/experiment_tracking/.env`. Notice there are two sets of credentials; use the ones for PgAdmin4. After authentication, you will see a screen like the one below.

<div><img src="./images/01_pgadmin_initial.png" height=450></div>

+ Click on "Add New Server":

    - In the *General* Tab, under Name enter: localhost. 
    - Under the *Connection* Tab, use Host name *postgres* (this is the name of the service in the docker compose file). 
    - Username and password are the ones found in the `./05_src/experiment_tracking/.env` file.


## Connect to MinIO

+ The interface for MinIO can be reached via [http://localhost:9001](http://localhost:9001)
+ The credentials can be found in the `./05_src/experiment_tracking/.env` file.

<div><img src="./images/01_minio.png" height=450></div>


## Learn More

+ Containers and containerization are topics well beyond the scope of this course. However, we will use containerized applications to help us implement certain patterns. 
+ If you are interested in Docker, a good place to start is the [Official Docker Guides](https://docs.docker.com/get-started/overview/).