# experiment_tracking

Docker Compose stack for local ML experiment tracking. Runs three services:

| Service | Image | Default port | Purpose |
|---------|-------|-------------|---------|
| `postgres` | `postgres:18` | `5432` | MLflow backend store |
| `pgadmin` | `dpage/pgadmin4:9.18` | `5051` | Web UI for PostgreSQL |
| `mlflow` | built from `./mlflow` | `5001` | MLflow tracking server and artifact proxy |

Artifacts are written to `./mlflow_artifacts/` on the host, which is bind-mounted into the `mlflow` container at `/mlflow/artifacts`. The server runs with `--serve-artifacts`, so clients upload and download artifacts over HTTP through `http://localhost:5001` and never need direct access to the storage.

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

---

## Starting the stack

```bash
# From this directory
docker compose up -d
```

Startup order is enforced by health conditions:
1. `postgres` starts and passes its healthcheck (`pg_isready`).
2. `mlflow` starts only after `postgres` is healthy.

First start may take 30–60 seconds for all services to be ready.

## Stopping the stack

```bash
docker compose down          # stop and remove containers, keep data
docker compose down -v       # also delete named volumes (destructive)
```

`postgres_data/` and `mlflow_artifacts/` are bind mounts, so `down -v` does not delete them. To start from a clean slate, stop the stack and delete both folders.

---

## Accessing the services

| Service | URL | Credentials |
|---------|-----|-------------|
| MLflow UI | http://localhost:5001 | — |
| pgAdmin | http://localhost:5051 | `PGADMIN_DEFAULT_EMAIL` / `PGADMIN_DEFAULT_PASSWORD` |
| PostgreSQL | `localhost:5432` | `POSTGRES_USER` / `POSTGRES_PASSWORD` |

Artifacts can be browsed in the MLflow UI (a run's *Artifacts* tab) or directly in `./mlflow_artifacts/`.

---

## Testing the connection

`test_mlflow.py` trains a small logistic regression model and logs it to the local MLflow server. Run it from the `05_src/` directory so that `utils.logger` is on the path:

```bash
# From 05_src/
uv run python experiment_tracking/test_mlflow.py
```

On success the script logs a `score` metric and a `model` artifact, then prints the run ID. Check the result at http://localhost:5001 under the `mlflow_test_experiment` experiment; the model files appear under `./mlflow_artifacts/`.

---

## Directory structure

```
experiment_tracking/
├── docker-compose.yml
├── .env                  # git-ignored — create from .env.example
├── .env.example          # committed reference with placeholder values
├── .gitattributes        # enforces LF line endings on *.sh files
├── mlflow/
│   ├── Dockerfile        # python:3.11-slim-bookworm + mlflow + psycopg2
│   └── requirements.txt
├── postgres/
│   └── init.sql          # creates the mlflow database on first start
├── mlflow_artifacts/     # git-ignored bind mount: MLflow artifact store
├── postgres_data/        # git-ignored bind mount: PostgreSQL data
└── test_mlflow.py        # smoke test
```
