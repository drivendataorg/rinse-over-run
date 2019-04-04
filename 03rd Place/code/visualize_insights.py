import shap
import matplotlib.pyplot
import logging as logger


def plot_shap(model, data, dependence_predictor=None, interaction='auto', cutoff=None):
    """Generate SHAP plots from trained model and validation or test set data.

    By default, this function will only generate a summary plot, which shows total variable importances.
    It can also generate dependence plots for specific predictors if the correct arguments are passed.
    These dependence plots will automatically display the second-order interaction that is estimated to be the
        strongest, unless the interaction arg is changed to None.

    Args:
        model (lightgbm.Booster): a lightgbm model.
        data (dict): dictionary with two keys, 'train' and 'eval', and two values corresponding to the training and
            evaluation datasets (lightgbm.Dataset objects).
        dependence_predictor (str): name of column to create dependence plot for. If None (default value), function
            call will not generate a dependence plot.
        interaction (str): value of 'interaction_index' arg for dependence plot. The value of 'auto' will automatically
            choose a second predictor that SHAP estimates has the strongest interaction with the primary predictor
            specified by the 'vis_dependence_predictor' arg and illustrate the interaction using color, while None will
            not show any interactions and all data points will be blue.
        cutoff (numeric): x-axis cutoff value for SHAP dependence plot. Useful if the dependence plot predictor has
            a small number of outliers that drastically skew the look of the plot.

    Returns:
        N/A.
    """

    matplotlib.pyplot.figure()
    explainer = shap.TreeExplainer(model)
    shap_data = data['eval'].data.copy()

    # If an x-axis cutoff is specified for the dependence plot predictor, subset the data accordingly
    if dependence_predictor is not None and cutoff is not None:
        shap_data = shap_data[shap_data[dependence_predictor] < cutoff]

    shap_values = explainer.shap_values(shap_data)

    # Create dependence plot for chosen predictor, if desired
    if dependence_predictor is not None:
        logger.info('Plotting SHAP dependence plot for predictor ' + dependence_predictor + '...')
        shap.dependence_plot(dependence_predictor, shap_values, shap_data, interaction_index=interaction)
        matplotlib.pyplot.figure()

    # Create summary plot of variable importances
    logger.info('Plotting SHAP summary plot (variable importances)...')
    shap.summary_plot(shap_values, shap_data, plot_type='bar', max_display=500)
    logger.info('SHAP plots created successfully.')
