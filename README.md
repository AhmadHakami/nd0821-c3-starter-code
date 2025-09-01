Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Environment Set up
* Recommended Python: 3.8 (the starter/requirements.txt targets Python 3.8).
* Create a conda environment or use your preferred environment manager.

Suggested conda command:

conda create -n ml-dev python=3.8 && conda activate ml-dev

Install dependencies (from the starter requirements file):

pip install -r starter/requirements.txt

* Install git if needed (e.g., sudo apt-get install git).

## Repositories
* Create a directory for the project and initialize git.
  * Commit changes frequently. Data and large trained artifacts are stored under starter/model in this repository for the exercise.
* Connect your local git repo to GitHub.
* Setup GitHub Actions to run tests (pytest) and linting (flake8) on push.

# Data
* Raw and cleaned data paths in the repo:
  - Raw (messy) data: starter/data/census.csv
  - Cleaned data: starter/data/census_clean.csv
* The training script reads starter/data/census_clean.csv by default.

# Model
* Training script (module): python -m starter.starter.train_model
  - This trains the model and saves artifacts to starter/model/ (model.joblib, encoder.joblib, lb.joblib, metrics.json, slice_output.txt).
* Model utilities and data processing are in starter/starter/ml/
* Metrics and per-slice output are saved to starter/model/metrics.json and starter/model/slice_output.txt

# Tests
* Unit tests are in starter/tests. Run them from the repository root:

pytest -q

# API Creation
* API implementation: starter/main.py (FastAPI)
  * Root GET returns a welcome message.
  * POST endpoint performs model inference using saved artifacts in starter/model/.
* Run the API locally:

uvicorn starter.main:app --reload --port 8000

* For production-style serving (Heroku or similar), use Gunicorn with Uvicorn worker:

gunicorn -k uvicorn.workers.UvicornWorker starter.main:app

# API Deployment
* Create a Heroku (or other) app and deploy from your GitHub repository.
  * Enable automatic deployments that only deploy if CI passes.
  * Configure environment variables and storage access as needed for your deployment.
* A simple script using the requests module can POST to the live API for smoke testing.

# Notes
* The repository contains a model card template at starter/model_card_template.md.
* If you change Python version, ensure CI and the starter/requirements.txt are kept consistent.
