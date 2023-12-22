import logging
from datetime import datetime
from logging.config import dictConfig
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from logging_config.logger_config import get_logger_config

dictConfig(get_logger_config())


def create_summary_plot(shap_values, feature_cols, model_infos, num_features=10, save=False, stage=1):
    title = (
        f"{model_infos['model_name']} - {model_infos['depvar']} - model_id {model_infos['model_id']} - stage {stage}"
    )
    num_feat = np.minimum(num_features, len(feature_cols) + 1)
    shap.summary_plot(
        shap_values, plot_type="bar", feature_names=feature_cols, max_display=num_feat, plot_size=0.15, show=False
    )
    plt.title(title)
    plt.tight_layout()
    if save:
        model_type = model_infos["model_name"].split("_")[-1]
        date = datetime.now().strftime("%Y%m%d")
        plt.savefig(
            f"./images/{model_infos['depvar']}_{model_type}/summary_plot_{title} - {num_feat}_features_{date}.png"
        )
    plt.show()


def explain_single_prediction(
    stages: List[int],
    models,
    model_infos: dict,
    dat: pd.DataFrame,
    station_id: str,
    include_split: bool,
    include_waterfall: bool,
    save: bool = False,
):
    dat = dat.query(f"station_id == '{station_id}'").copy(deep=True)

    title_1 = f"{model_infos['model_name']} - {model_infos['depvar']} - model_id {model_infos['model_id']}"
    date = dat.date_time.item().tz_localize(tz="UTC").tz_convert(tz="Europe/Berlin").strftime("%Y-%m-%d %H:%M:%S")
    title_2 = f"STAGE \n {date} - station '{station_id}'"
    title = f"{title_1} - {title_2}"

    if include_split:
        create_split_plot(models, model_infos, dat, title, save)

    if include_waterfall:
        create_waterfall_plot(stages, models, model_infos, dat, title, save)


def create_waterfall_plot(stages, models, model_infos, dat, title, save):
    explainer_1 = shap.TreeExplainer(models.model_1)

    xtrain = dat.loc[:, models.feature_cols_1]
    labels = dat.loc[:, models.depvar]

    dmatrix_1 = xgb.DMatrix(xtrain, label=labels, enable_categorical=True, feature_names=models.feature_cols_1)

    if 1 in stages:
        title_1 = title.replace("STAGE", "stage 1")

        explanation = explainer_1(dmatrix_1)
        exp = shap.Explanation(
            explanation.values[0], explanation.base_values[0], xtrain.values[0], feature_names=models.feature_cols_1
        )
        shap.plots.waterfall(exp, show=False)
        _, h = plt.gcf().get_size_inches()
        plt.gcf().set_size_inches(7, 4)
        plt.title(title_1)
        plt.tight_layout()
        if save:
            temp = title_1.split("\n")
            title_1 = f"{temp[0]} - {temp[1]}"
            model_type = model_infos["model_name"].split("_")[-1]
            date = datetime.now().strftime("%Y%m%d")
            plt.savefig(f"./images/{model_infos['depvar']}_{model_type}/waterfall_plot_{title_1}_{date}.png")
        plt.show()

    if 2 in stages:
        if not models.use_two_stages:
            logging.warning("Can't explain second stage if the model has only one.")

        else:
            title_2 = title.replace("STAGE", "stage 2")

            prediction = models.model_1.predict(dmatrix_1)
            residual = dat[models.depvar] - prediction

            xtrain = dat.loc[:, models.feature_cols_2]
            labels = residual

            dmatrix_2 = xgb.DMatrix(xtrain, label=labels, enable_categorical=True, feature_names=models.feature_cols_2)

            explainer_2 = shap.TreeExplainer(models.model_2)
            explanation = explainer_2(dmatrix_2)
            exp = shap.Explanation(
                explanation.values[0], explanation.base_values[0], xtrain.values[0], feature_names=models.feature_cols_2
            )
            shap.plots.waterfall(exp, show=False)
            plt.gcf().set_size_inches(8, 4)
            plt.title(title_2)
            plt.tight_layout()
            if save:
                temp = title_2.split("\n")
                title_2 = f"{temp[0]} - {temp[1]}"
                model_type = model_infos["model_name"].split("_")[-1]
                date = datetime.now().strftime("%Y%m%d")
                plt.savefig(f"./images/{model_infos['depvar']}_{model_type}/waterfall_plot_{title_2}_{date}.png")
            plt.show()


def create_split_plot(models, model_infos, dat, title, save=False):
    title = title.replace("STAGE", "")

    prediction_total, prediction_1, prediction_residual = models.predict(dat)
    observation = dat[models.depvar]
    plt.figure(figsize=(10, 4))
    plt.barh("Stage1", prediction_1)
    if models.use_two_stages:
        plt.barh("Stage2", prediction_residual)
    plt.barh("Prediction", prediction_total)
    plt.barh("Observation", observation)
    plt.vlines(x=observation, ymin=-1, ymax=4, linestyle="dashed", color="black")
    plt.grid(axis="x")
    plt.title(title)
    if save:
        temp = title.split("\n")
        title = f"{temp[0]} - {temp[1]}"
        model_type = model_infos["model_name"].split("_")[-1]
        date = datetime.now().strftime("%Y%m%d")
        plt.savefig(f"./images/{model_infos['depvar']}_{model_type}/prediction_split_plot_{title}_{date}.png")
    plt.show()


def explain_summary_plot(
    stages: List[int], models, dat: pd.DataFrame, model_infos: dict, num_features_max: int = 10, save: bool = False
):
    # Create dmatrix
    dmatrix_1 = xgb.DMatrix(
        dat.loc[:, models.feature_cols_1],
        label=dat.loc[:, models.depvar],
        enable_categorical=True,
        feature_names=models.feature_cols_1,
    )

    if 1 in stages:
        # Create first shapley summary
        explainer_1 = shap.TreeExplainer(models.model_1)
        shap_values = explainer_1.shap_values(dmatrix_1)
        create_summary_plot(
            shap_values,
            models.feature_cols_1,
            model_infos,
            num_features=num_features_max,
            save=save,
            stage=1,
        )

    if 2 in stages:
        if not models.use_two_stages:
            logging.warning("Can't explain second stage if the model has only one.")

        else:
            # Calculate predictions of first stage
            prediction = models.model_1.predict(dmatrix_1)
            residual = dat[models.depvar] - prediction

            # Create dmatrix
            dmatrix_2 = xgb.DMatrix(
                dat.loc[:, models.feature_cols_2],
                label=residual,
                feature_names=models.feature_cols_2,
                enable_categorical=True,
            )
            # Create second shapley summary
            explainer_2 = shap.TreeExplainer(models.model_2)
            shap_values = explainer_2.shap_values(dmatrix_2)
            create_summary_plot(
                shap_values,
                models.feature_cols_2,
                model_infos,
                num_features=num_features_max,
                save=save,
                stage=2,
            )


def explain_dependence_plots(
    models, model_infos: dict, dat: pd.DataFrame, feature, alpha=0.5, save=False, cutoff=1, dot_size=16
):
    if feature in models.feature_cols_1:
        stage = 1
    elif feature in models.feature_cols_2:
        stage = 2
    else:
        logging.warning(f"Selected feature {feature} isn't used in stage 1 nor stage 2.")
        return

    title = (
        f"{model_infos['model_name']} - {model_infos['depvar']} - model_id {model_infos['model_id']} - stage {stage}"
    )
    model_type = model_infos["model_name"].split("_")[-1]

    # Create dmatrix
    dmatrix_1 = xgb.DMatrix(
        dat.loc[:, models.feature_cols_1],
        label=dat.loc[:, models.depvar],
        enable_categorical=True,
        feature_names=models.feature_cols_1,
    )

    # Calculate shap values
    if stage == 1:
        explainer_1 = shap.TreeExplainer(models.model_1)
        explanation_data_1 = explainer_1(dmatrix_1)
        shap_values = explanation_data_1.values
        dat_data = dat.loc[:, models.feature_cols_1]

    if stage == 2:
        # Calculate predictions of first stage
        prediction = models.model_1.predict(dmatrix_1)
        residual = dat[models.depvar] - prediction

        # Create dmatrix
        dmatrix_2 = xgb.DMatrix(
            dat.loc[:, models.feature_cols_2],
            label=residual,
            feature_names=models.feature_cols_2,
            enable_categorical=True,
        )

        explainer_2 = shap.TreeExplainer(models.model_2)
        explanation_data_2 = explainer_2(dmatrix_2)
        shap_values = explanation_data_2.values

        dat_data = dat.loc[:, models.feature_cols_2]

    # Create dependence plot
    shap.dependence_plot(
        feature,
        shap_values,
        dat_data,
        xmin=f"percentile({cutoff})",
        xmax=f"percentile({100-cutoff})",
        interaction_index=None,
        alpha=alpha,
        dot_size=dot_size,
        show=False,
    )
    plt.title(title)
    plt.tight_layout()
    if save:
        date = datetime.now().strftime("%Y%m%d")
        plt.savefig(f"./images/{model_infos['depvar']}_{model_type}/dep_plot_{title} - {feature}_{date}.png")
    plt.show()
