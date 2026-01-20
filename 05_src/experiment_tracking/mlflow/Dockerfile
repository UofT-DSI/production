FROM python:3.11-slim-buster

# Install python package
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Expose MLflow port
EXPOSE 5000

# Set working directory
WORKDIR /mlflow

# Run MLflow server (this will be overridden by docker-compose command)
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]