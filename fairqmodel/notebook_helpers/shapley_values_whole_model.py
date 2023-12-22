# NOTE: This script is similar to shapley_values.py, but uses a different explainer
# and requires slightly different handling.
# For better readability the functions are not further parameterized but re-defined, accepting some redundancy.

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import OrdinalEncoder

from fairqmodel.data_preprocessing import fix_column_types
from fairqmodel.model_wrapper import ModelWrapper


class prediction_helper:
    def __init__(
        self,
        models: ModelWrapper,
        encoder: OrdinalEncoder,
        cat_cols: List[str],
        categorical_feature_cols: List[str],
        metric_feature_cols: List[str],
    ):
        """Helper class to enable the calculation of shapley values for the two-staged model.
        This is necessary because:
            - shap.Explainer (different to shap.TreeExplainer) can't handel categorical inputs
            - the models.predict() function has more than one return value:
                    (prediction_total, prediction_stage_1, prediction_stage_2)

        :param models: ModelWrapper, Entire model
        :param encoder: OrdinalEncoder, Required to decode encoded categorical variables
        :param cat_cals: List[str], Names of variables that have been encoded
        :param categorical_feature_cols: List[str], Names of the columns that need to be of type 'categorical'
        :param metric_feature_cols: List[str], Names of the columns that need to be of type 'float'

        """
        self.models = models
        self.encoder = encoder
        self.cat_cols = cat_cols
        self.categorical_feature_cols = categorical_feature_cols
        self.metric_feature_cols = metric_feature_cols

    def call_prediction(self, df: pd.DataFrame) -> np.ndarray:
        """Calls the predict function, including the remaining parameters.
        In a first step, the encoded categorical variables have to be decoded.

        :param df: pd.DataFrame, Containing all data, but categorical variables are encoded

        :return: np.ndarray, Predicted values
        """
        df_copy = df.copy(deep=True)  # required to avoid 'copy to slice of df'
        df_copy.loc[:, self.cat_cols] = self.encoder.inverse_transform(df.loc[:, self.cat_cols])  # Decode categorical
        df_copy = fix_column_types(df_copy, self.categorical_feature_cols, self.metric_feature_cols)  # Correct dtype
        all_predictions, _, _ = self.models.predict(df_copy)  # Perform prediction

        return all_predictions


def create_summary_plot(explanation: shap.Explanation, feature_cols: list[str], num_features: int = 10) -> None:
    """Creates a plot summarizing pre-computed influences that each variable has on the total prediction
    :param explanation: shap.Explanation, Containing the shap values
    :param feature_cols: list[str], Containing the names of the independent variables
    :param num_features: int, Specifies the number of features explained in the plot
    :return: None
    """
    title = "Summary Plot for entire model"

    num_feat = np.minimum(num_features, len(feature_cols) + 1)

    # Create the plot
    shap.summary_plot(
        explanation,
        plot_type="bar",
        feature_names=feature_cols,
        max_display=num_feat,
        plot_size=(10, num_feat // 2),
        show=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


def create_dependence_plot(
    explanation: shap.Explanation,
    feature: str,
    df: pd.DataFrame,
    interaction_feature: Optional[str] = None,
    alpha: float = 0.7,
    dot_size: int = 16,
    plot_width: int = 6,
    plot_height: int = 4,
) -> None:
    """Creates the dependence plot for the selected feature
    :param explanation: shap.Explanation, Containing the shap values
    :param feature: str, Name of the feature
    :param df: pd.DataFrame, Containing the data
    :param interaction_feature: Optional[str], Name of the feature to show interactions for
    :param alpha: float, Controls the transparency of each dot in the plot (for readability)
    :param dot_size: int, Controls the size of each dot in the plot (for readability)
    :param plot_width: int, Width of the plot
    :param plot_height:int, Height of the plot
    :return: None
    """
    title = f"Dependence Plot for {feature}"

    # Create the plot
    shap.dependence_plot(
        feature,
        explanation.values,
        df,
        interaction_index=interaction_feature,
        alpha=alpha,
        dot_size=dot_size,
        show=False,
    )
    # Specify the layout of the plot
    _, h = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(plot_width, plot_height)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def create_waterfall_plots(
    explanation: shap.Explanation,
    df: pd.DataFrame,
    n_features: int = 10,
    x: int = 0,
) -> None:
    """Creates the waterfall plot for a selected customer.
    The plot shows how the final prediction was influenced by each variable.
    :param explanation: shap.Explanation, Containing the shap values
    :param df: pd.DataFrame, Containing the data
    :param n_features: int, Number of shown features
    :param x: int, Specifies the row to explain

    :return: None
    """
    station_id = df.loc[x:x, :].station_id
    date_time = df.loc[x:x, :].date_time
    title = f"Waterfall Plot for station_id: {station_id}, date: '{date_time}'"

    # Create the plot
    exp = shap.Explanation(
        explanation.values[x],
        explanation.base_values[x],
        df.values[x],
        feature_names=df.columns,
    )
    shap.plots.waterfall(exp, show=False, max_display=n_features)

    # Layout the plot
    _, h = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(10, int(0.4 * n_features))

    plt.title(title)
    plt.tight_layout()

    plt.show()
