import logging
from logging.config import dictConfig

import numpy as np
import optuna
from optuna.samplers import TPESampler

from fairqmodel.build_splits import prepare_folded_input
from fairqmodel.data_preprocessing import cap_high_values
from fairqmodel.db_connect import db_connect_target, get_query
from fairqmodel.feature_selection import assign_features_to_stage
from fairqmodel.model_parameters import get_xgboost_param
from fairqmodel.model_wrapper import ModelWrapper
from fairqmodel.prediction_lag_adjusted import make_lag_adjusted_prediction
from fairqmodel.prediction_t_plus_k import get_model_settings
from fairqmodel.read_write_model_aux_functions import model_name_str
from fairqmodel.train_model_in_hpo import train_hpo
from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())


def hyper_opt(
    depvar,
    params,
    lag_options,
    lags_avg,
    dat,
    feature_cols,
    num_boost_round=500,
    early_stopping_rounds=20,
    max_minutes=10,
    n_trials=None,
    study_name="XGBoost-HyperOpt",
    silence=False,
    n_cores=8,
    seed=123,
    t_plus_k_params=None,
    storage=False,
    use_lags=True,
    optimize_lags=False,
    use_two_stages=False,
    optimize_stage=None,
    first_stage_model_id=None,
):
    """Function to tune hyper-parameters using optuna.

    Parameters
    ----------
    depvar: str
        name of dependent variable
    params: dict
        Booster params in the form of "params_name": [min_val, max_val].
    lag_options: list[list[int]]
        list of lag combinations to choose from
    lags_avg: list[int]
        Contains lags for avg feature
    dat: pd.DataFrame
        Data for training, testing and evaluation
    num_boost_round : int
        Number of boosting iterations.
    early_stopping_rounds: int
        Activates early stopping. Cross-Validation metric (average of validation
        metric computed over CV folds) needs to improve at least once in
        every **early_stopping_rounds** round(s) to continue training.
        The last entry in the evaluation history will represent the best iteration.
        If there's more than one metric in the **eval_metric** parameter given in
        **params**, the last metric will be used for early stopping.
    max_minutes : int
        Time budget in minutes, i.e., stop study after the given number of minutes.
    n_trials : Optional[int]
        The number of trials. If this argument is set to None, there is no limitation on the number of trials.
    study_name : str
        Name of the hyperparameter study.
    silence : bool
        Controls the verbosity of the trail, i.e., user can silence the outputs of the trail.
    t_plus_k_params: Optional[dict]
        Parameters for the t+k CV
    use_lags: bool
        specifies if lags are included in optimization
    optimize_lags: bool
        specifies if lags are optimized
    use_two_stages: bool,
        Specifies if the first or a second stage is optimized.
    optimize_stage: Optional[int],
        If set, specifies which stage should be optimized.
    first_stage_model_id: Optional[int],
        Specifies the model to use as a first stage, when optimizing second stage.
        Otherwise this variable has no effect.
    Returns
    -------
    opt_params : Dict() with optimal parameters.
    """

    if storage:
        # TODO: set up real db instead of creating local optuna_hpo_studies.db file on the fly
        storage_name = "sqlite:///optuna_hpo_studies.db"
    else:
        storage_name = None

    def objective(trial):
        hyper_params = {
            "booster": "gbtree",
            "eta": trial.suggest_loguniform(
                "eta",
                params["eta"][0],
                params["eta"][1],
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                params["max_depth"][0],
                params["max_depth"][1],
            ),
            "gamma": trial.suggest_loguniform(
                "gamma",
                params["gamma"][0],
                params["gamma"][1],
            ),
            "subsample": trial.suggest_loguniform(
                "subsample",
                params["subsample"][0],
                params["subsample"][1],
            ),
            "colsample_bytree": trial.suggest_loguniform(
                "colsample_bytree",
                params["colsample_bytree"][0],
                params["colsample_bytree"][1],
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                params["min_child_weight"][0],
                params["min_child_weight"][1],
            ),
            "nthread": n_cores,
        }
        if optimize_stage == 1:
            hyper_params["monotone_constraints"] = {"kfz_per_hour": 1}
            print(hyper_params)

        if use_lags:
            if optimize_lags:
                lag_idx = trial.suggest_int("lag_setting", params["lag_idx"][0], params["lag_idx"][1])
            else:
                lag_idx = 0
        else:
            lag_idx = None

        return get_best_score(
            depvar,
            trial,
            lag_options,
            lags_avg,
            dat,
            feature_cols,
            lag_idx,
            hyper_params,
            num_boost_round,
            early_stopping_rounds,
            t_plus_k_params,
            use_two_stages,
            optimize_stage,
            first_stage_model_id,
        )

    if silence:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,  # if the same study name exists, load it and continue
    )

    # Add sensible default trial to speed up HPO
    # Start with the currently best set of parameters
    params_previous, _ = get_xgboost_param(depvar, use_two_stages, stage=optimize_stage, dev=False)

    study.enqueue_trial(
        {
            "eta": params_previous["eta"],
            "max_depth": params_previous["max_depth"],
            "gamma": params_previous["gamma"],
            "min_child_weight": params_previous["min_child_weight"],
            "subsample": params_previous["subsample"],
            "colsample_bytree": params_previous["colsample_bytree"],
        }
    )

    timeout = 60 * max_minutes
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
    )

    logging.info("Hyper-Parameter Optimization successfully finished.")
    logging.info(f"Number of finished trials: {len(study.trials)}")
    logging.info("Best trial:")
    opt_param = study.best_trial

    # Add optimal stopping round
    opt_param.params["opt_rounds"] = study.trials_dataframe()["user_attrs_opt_round"][
        study.trials_dataframe()["value"].idxmin()
    ]
    opt_param.params["opt_rounds"] = int(opt_param.params["opt_rounds"])

    logging.info("\t\t Value: {}".format(opt_param.value))
    logging.info("\t\t Params: ")
    for key, value in opt_param.params.items():
        logging.info("\t\t {}: {}".format(key, value))

    return {"study": study, "opt_param": opt_param.params}


def get_best_score(
    depvar,
    trial,
    lag_options,
    lags_avg,
    dat,
    feature_cols,
    lag_idx,
    hyper_params,
    num_boost_round,
    early_stopping_rounds,
    t_plus_k_params,
    use_two_stages,
    optimize_stage,
    first_stage_model_id,
):
    # remove all not selected lags from feature cols
    feature_cols = [feature for feature in feature_cols if "lag" not in feature]

    # add the lags selected for this trial (if any)
    lags_actual = []
    if len(lag_options) > 0:
        lags_actual = lag_options[lag_idx]
        lag_vars = [f"{depvar}_lag{lag}" for lag in lags_actual]
        feature_cols.extend(lag_vars)

    best_score, best_iteration = manual_cv_t_plus_k(
        dat,
        feature_cols,
        depvar,
        hyper_params,
        lags_actual,
        lags_avg,
        num_boost_round,
        early_stopping_rounds,
        t_plus_k_params,
        use_two_stages,
        optimize_stage,
        first_stage_model_id,
    )

    trial.set_user_attr("opt_round", best_iteration)

    return best_score


def manual_cv_t_plus_k(
    dat,
    feature_cols,
    depvar,
    xgb_param,
    lags_actual,
    lags_avg,
    n_rounds,
    early_stopping_rounds,
    t_plus_k_params,
    use_two_stages,
    optimize_stage,
    first_stage_model_id,
):
    time_cv_folds = prepare_folded_input(
        dat,
        n_cv_windows=t_plus_k_params["n_cv_windows"],
        n_train_years=t_plus_k_params["n_train_years"],
        test_cv_window_size=t_plus_k_params["test_cv_window_size"],
        step_size=t_plus_k_params["step_size"],
        prediction_hour=t_plus_k_params["prediction_hour"],
    )

    prediction_metrics = []
    best_iterations = []

    # Assign features to model stages
    features_stage_1, features_stage_2 = assign_features_to_stage(use_two_stages, feature_cols)

    loop_idx = 0
    for fold in reversed(time_cv_folds):
        train_model = (loop_idx) % t_plus_k_params["train_shift"] == 0

        if train_model:
            # Update 'max_train_date' only if model is actually trained
            max_train_date = fold["ts_fold_max_train_date"].strftime("%Y-%m-%d %H:%M:%S")

        max_test_date = fold["ts_fold_max_test_date"].strftime("%Y-%m-%d %H:%M:%S")

        logging.info(f"Fold {loop_idx+1}/{len(time_cv_folds)}, train_model: {train_model}")
        logging.info(f"max_train_date: {max_train_date}, max_test_date: {max_test_date}")

        # Get the test data
        dat_test = fold["test"]

        # Split the train data into train and validation, where the validation contains
        time_delta = dat_test.date_time.max() - dat_test.date_time.max()
        dat_train, dat_val = split_train_data(fold["train"], time_delta)

        if train_model:
            # Load first model if second stage is optimized
            if optimize_stage == 2:
                logging.info("Load first stage")
                query_params = {"model_type": model_name_str("all"), "depvar": depvar}
                with db_connect_target() as db:
                    available_models = db.query_dataframe(get_query("available_models"), params=query_params)
                assert first_stage_model_id is not None
                model_settings = get_model_settings(available_models, first_stage_model_id)
                first_stage_model = model_settings["models"].model_1
            else:
                first_stage_model = None

            # Train model(s) with the suggested parameters
            model_1, model_2 = train_hpo(
                dat=cap_high_values(dat_train.copy(deep=True), depvar),
                dat_early_stopping=dat_val,
                depvar=depvar,
                feature_cols_1=features_stage_1,
                feature_cols_2=features_stage_2,
                xgb_param=xgb_param,
                n_rounds=n_rounds,
                early_stopping_rounds=early_stopping_rounds,
                early_stopping_threshold=0.05,
                first_stage_model=first_stage_model,
            )
            # Initialize model wrapper instance with trained models
            models = ModelWrapper(
                depvar=depvar,
                model_1=model_1,
                model_2=model_2,
            )

        # Evaluate the model
        _, rmse = make_lag_adjusted_prediction(
            fold, models, feature_cols, depvar, lags_actual, lags_avg, calc_metrics=True
        )

        prediction_metrics.append(rmse)

        best_it = models.model_1.best_iteration if optimize_stage == 1 else models.model_2.best_iteration
        best_iterations.append(best_it + 1)

        loop_idx += 1

    avg_rmse = np.mean(prediction_metrics)
    best_iteration = int(np.quantile(best_iterations, 0.75))

    logging.info(f"Finish iteration with avg_rmse: {avg_rmse=:.3f} and best_iteration: {best_iteration}\n\n")

    # NOTE: Using the mean over the best iterations is the best approximation to the true value
    # as the values needed to calculate the exact best iteration are not available.
    return avg_rmse, best_iteration


def split_train_data(dat_all, time_delta):
    max_test_date = dat_all.date_time.max()
    min_test_date = max_test_date - time_delta

    dat_train = dat_all[dat_all["date_time"] < min_test_date]
    dat_val = dat_all[dat_all["date_time"] >= min_test_date]

    return dat_train, dat_val
