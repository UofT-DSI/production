# docker/minio/create-bucket.sh
#!/bin/sh
# Configure MinIO Client
until (/usr/bin/mc config host add minio http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}); do
  echo "Waiting for MinIO to be ready..."
  sleep 1
done
mc alias set minioserver http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}

# Create the MLFlow bucket
mc mb minioserver/mlflow