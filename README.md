# fairq-model
This repo contains the Python code of the entire life cycle of the air quality model, including:

- Hyperparameter optimization
- Model training
- Scripts for multiple prediction tasks
- Model evaluation

## Model
- The model is a XGB-Regression model predicting the pollotion level (in µg/m³).

- For each pollutant type a single model is used.
- A model can either consist of one or of two stages. In a one-stage model all features are  used simultaneously as inputs. In a two-stage model the first stage uses only traffic and time-constant variables to make an initial prediction and the second stage learns to predict the resulting residuals. This procedure allows for easier simulation of traffic reductions.

## How to get started
- Create an `.env` file in the project folder, see env_template for the structure (Note: The database containing all features such as the weather and traffic data must be created previously, see https://github.com/fairqBerlin/fairq-data/tree/public/inst/db.)
- Install required packages from the Pipfile by running
```shell
pipenv install
```
- Activate the local environment with
```shell
pipenv shell
```

## Executable Scripts
In the following the most important scripts are outlined, each with a short instruction on how to use them.
Note that every script has selectable parameters that can be specified via the command line or directly in the script.
General parameters are:

- `depvar`: Name of the dependent variable, i.e. the type of pollutant (either "no2", "pm10" or "pm25")
- `use_lags`: Specifies if time-lagged values of the target variable are included as a feature
- `use_two_stages`: Specifies if the model is comprised of one or two stages (see description of the model above)
- `write_db`: Specifies if results are written to the database


### Hyperparameter optimization [HPO]
HPO is implemented using the [optuna](https://github.com/optuna/optuna) package.

The relevant code can be found in `1_run_hyper_optim.py`.

Start the hyperparameter optimization with:
```shell
pipenv run python 1_run_hyper_optim.py
```

The HPO trials are saved to a file named `optuna_hpo_studies.db`.

You can monitor the running trials by starting the optuna dashboard with this file:
```shell
pipenv run optuna-dashboard sqlite:///optuna_hpo_studies.db  --port 8080
```

### Model Training
The model training script can be used to train a new model, e.g. when new data is available.
The relevant code can be found in `2_train.py`.
Start the training by running:
```shell
pipenv run python 2_train.py
```
You might want to specify some of the selectable parameters described above.
Trained models are written to the database in JSON format (schema fairq\_(prod\_)output).

### Predictions
All scripts that perform predictions have the prefix '3_'.
Predictions can be made for different settings:
- `3a_make_pred_on_grid`: Creates predictions for every cell of a 50x50m² grid in Berlin
- `3b_make_pred_at_stations_past`: Creates predictions at the measuring stations in the past
- `3c_make_pred_at_station_future`: Creates predictions at the measuring stations for the future. The maximal forecast horizon is limited by the availability of the input variables as e.g. the weather data/forecasts. The usual forecast horizon is four days.
- `3d_make_pred_at_station_kfz_adjusted`: Creates predictions at the measuring stations including simulated changes of the traffic-intensity
- `3e_make_predictions_at_passive_samplers`: Creates predictions at the locations of passive samples

These scripts can be executed by running:
```shell
pipenv run python <filename> <selectable_parameters>
```
Predictions are written to the database in JSON format (schema fairq\_(prod\_)output).

### Model Evaluation
The model (parameter) evaluation is performed by cross validation.


To estimate how good the model generalizes in the spatial domain, a 'leave-one-station-out' approach is used. The model is trained on the data of all but one stations. The evaluation is performed on the left out station. This procedure is repeated for every station and the results are averaged.
This evaluation can be executed by running:
```shell
pipenv run python 4_cv_loso.py
```

To estimate how good the models predictions into the future are, a temporally resolved cross validation is performed. It can be executed by:
```shell
pipenv run python 5_cv_time_t_plus_k.py
```
The results are written to the database in JSON format (schema fairq\_(prod\_)output) to be available for further model evaluation using dashboards.

## Style checking

You can run style checking commands in the console, which we usually do automatically via Jenkins (not published). This sections lists the checks and tells how to fix problems.

### mypy static type enforcement
- Check: `mypy . --namespace-packages`
- Fix problems by fixing inconsistent typing

### Flake8 Styleguide Enforcement
- Check: `flake8 .`
- Fix the displayed problems manually in the files

### Black Styleguide Enforcement
- Check: `black --line-length 120 --check .`
- Fix problems using the black auto formatter:
  - Installation: File -> Settings -> Tools -> External Tools -> "+" -> add "Black"
    - Program: `~/.local/share/virtualenvs/fairq-data-cams-xxxx/bin/black` (adjust for your VE path)
    - Arguments: `$FilePath$ --line-length 120`
    - Working directory: `$ProjectFileDir$`
  - Add a shortkey to run the auto formatter, e.g., CTRL+SHIFT+A

### isort: order of import statements
- Fix problems via `isort .`


## Jupyter Notebooks (unpublished)
The Notebook folder contains several jupyter notebooks for visualization, including variable exploration, shapley plots to estimate feature importance and simulations of kfz-adjustments.
For jupyter notebooks to work, first install everything from the Pipfile and then run
`pipenv run jupyter contrib nbextension install --user`

