# experiment_tracking

Docker Compose stack for local ML experiment tracking. Runs four services:

| Service | Image | Default port | Purpose |
|---------|-------|-------------|---------|
| `postgres` | `postgres:17-trixie` | `5432` | MLflow backend store |
| `pgadmin` | `dpage/pgadmin4:9` | `5051` | Web UI for PostgreSQL |
| `minio` | `quay.io/minio/minio` | `9000` / `9001` | S3-compatible artifact store |
| `mlflow` | built from `./mlflow` | `5001` | MLflow tracking server |

`minio-setup` is a one-shot init container that creates the `mlflow` bucket. Its logic is inlined directly in `docker-compose.yml` — there is no external shell script.

---

## Prerequisites

- Docker Desktop (or Docker Engine + Compose plugin)
- A `.env` file in this directory (see [Configuration](#configuration) below)

---

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

`.env` is git-ignored and must never be committed. The variables it must define:

| Variable | Used by | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | postgres, mlflow | Database superuser name |
| `POSTGRES_PASSWORD` | postgres, mlflow | Database superuser password |
| `POSTGRES_DB` | postgres | Default database name |
| `PGADMIN_DEFAULT_EMAIL` | pgadmin | pgAdmin login e-mail |
| `PGADMIN_DEFAULT_PASSWORD` | pgadmin | pgAdmin login password |
| `MINIO_ACCESS_KEY` | minio, minio-setup, mlflow | MinIO root user / access key |
| `MINIO_SECRET_ACCESS_KEY` | minio, minio-setup, mlflow | MinIO root password / secret key |

---

## Starting the stack

```bash
# From this directory
docker compose up -d
```

Startup order is enforced by health conditions:
1. `postgres` starts and passes its healthcheck (`pg_isready`).
2. `minio` starts; `minio-setup` polls it with `mc alias set` until it accepts connections, then creates the `mlflow` bucket.
3. `mlflow` starts only after `postgres` is healthy and `minio-setup` has completed successfully.

First start may take 30–60 seconds for all services to be ready.

## Stopping the stack

```bash
docker compose down          # stop and remove containers, keep volumes
docker compose down -v       # also delete postgres and minio volumes (destructive)
```

---

## Accessing the services

| Service | URL | Credentials |
|---------|-----|-------------|
| MLflow UI | http://localhost:5001 | — |
| MinIO Console | http://localhost:9001 | `MINIO_ACCESS_KEY` / `MINIO_SECRET_ACCESS_KEY` |
| pgAdmin | http://localhost:5051 | `PGADMIN_DEFAULT_EMAIL` / `PGADMIN_DEFAULT_PASSWORD` |
| PostgreSQL | `localhost:5432` | `POSTGRES_USER` / `POSTGRES_PASSWORD` |

---

## Testing the connection

`test_mlflow.py` trains a small logistic regression model and logs it to the local MLflow server. Run it from the `05_src/` directory so that `utils.logger` is on the path:

```bash
# From 05_src/
uv run python experiment_tracking/test_mlflow.py
```

On success the script logs a `score` metric and a `model` artifact, then prints the run ID. Check the result at http://localhost:5001 under the `mlflow_test_experiment` experiment.

---

## Directory structure

```
experiment_tracking/
├── docker-compose.yml
├── .env                  # git-ignored — create from .env.example
├── .env.example          # committed reference with placeholder values
├── .gitattributes        # enforces LF line endings on *.sh files
├── mlflow/
│   ├── Dockerfile        # python:3.11-slim-bookworm + mlflow + boto3
│   └── requirements.txt
├── postgres/
│   └── init.sql          # creates the mlflow database on first start
├── minio_data/           # git-ignored volume mount
├── postgres_data/        # git-ignored volume mount
└── test_mlflow.py        # smoke test
```
