### Instructions

#### 1 Create and activate a Python environment
Use Python 3.8 (recommended). Example using conda:

conda create -n ml-dev python=3.8 && conda activate ml-dev

(Or use your preferred environment manager / virtualenv.)

#### 2 Install project dependencies
Install dependencies from the starter requirements file:

pip install -r starter/requirements.txt

If you prefer to install from the root requirements.txt, ensure it matches the starter/requirements.txt.

#### 3 Train the model
The training script is provided as a module. Run it from the repository root to train and save artifacts:

python -m starter.starter.train_model

This will read the cleaned data at starter/data/census_clean.csv and save artifacts to starter/model/ (model.joblib, encoder.joblib, lb.joblib, metrics.json, slice_output.txt).

#### 4 Run tests
Unit tests are under starter/tests. Run pytest from the repository root:

pytest -q

#### 5 Run the API locally
The FastAPI app is implemented in starter/main.py. To run locally for development:

uvicorn starter.main:app --reload --port 8000

API details:
- GET / returns a welcome message
- POST /predict (or the configured inference endpoint) accepts a JSON body matching the CSV column names and returns model predictions (the app loads artifacts from starter/model/)

For production-style serving use Gunicorn with Uvicorn worker:

gunicorn -k uvicorn.workers.UvicornWorker starter.main:app

#### 6 Notebook and data cleaning
The data cleaning notebook is data_cleaning_eda.ipynb. It reads the raw file at starter/data/census.csv and (in the notebook) writes starter/data/census_clean.csv after cleaning. If you update the notebook, re-run it and commit starter/data/census_clean.csv.

#### 7 Helpful notes
- Data paths used by the training script and API are inside starter/ (starter/data and starter/model).
- If you change Python versions, update CI and the requirements files accordingly.
- To quickly retrain after changes, run the training module again: python -m starter.starter.train_model