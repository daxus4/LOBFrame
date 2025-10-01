from data_processing.complete_homological_utils import get_complete_homology
from loggers import logger
from utils import load_yaml, parse_args

if __name__ == "__main__":
    # Parse input arguments.
    args = parse_args()
    wb_error_detection = False

    experiment_id = args.experiment_id

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

    get_complete_homology(
        general_hyperparameters=general_hyperparameters,
        model_hyperparameters=model_hyperparameters,
    )
