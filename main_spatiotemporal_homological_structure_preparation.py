from data_processing import data_process_utils
from data_processing.complete_homological_utils import get_complete_homology
from data_processing.spatiotemporal_utils.spatiotemporal_utils import (
    execute_spatiotemporal_tmfg_pipeline,
)
from loggers import logger
from optimizers.executor import Executor
from simulator import market_sim, post_trading_analysis
from utils import (
    create_hyperparameters_yaml,
    data_split,
    is_hyperparams_yaml_existing,
    load_yaml,
    parse_args,
    save_dataset_info,
)

if __name__ == "__main__":
    # Parse input arguments.
    args = parse_args()
    wb_error_detection = False

    experiment_id = args.experiment_id
    if not is_hyperparams_yaml_existing(experiment_id):
        raise ValueError(
            f"Experiment ID {experiment_id} does not exist. Please provide a valid experiment ID."
        )

    # Load the configuration file containing the hyperparameters.
    hyperparameters_path = (
        f"{logger.find_save_path(experiment_id)}/hyperparameters.yaml"
    )

    # Load the configuration file (general hyperparameters).
    general_hyperparameters = load_yaml(hyperparameters_path, "general")
    # Load the configuration file (model's hyperparameters).
    model_hyperparameters = load_yaml(hyperparameters_path, "model")
    # Load the configuration file (trading hyperparameters).
    trading_hyperparameters = load_yaml(hyperparameters_path, "trading")

    execute_spatiotemporal_tmfg_pipeline(
        general_hyperparameters=general_hyperparameters,
        model_hyperparameters=model_hyperparameters,
    )
