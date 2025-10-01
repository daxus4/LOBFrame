from loggers import logger
from optimizers.executor import Executor
from simulator import market_sim, post_trading_analysis
from utils import is_hyperparams_yaml_existing, load_yaml, parse_args, save_dataset_info

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

    general_hyperparameters["stages"] = args.stages.split(",")

    # Instantiate the executor as None.
    executor = None
    # For 'torch_dataset_preparation' stage, instantiate the executor with proper arguments.
    if "torch_dataset_preparation" in general_hyperparameters["stages"]:
        executor = Executor(
            experiment_id,
            general_hyperparameters,
            model_hyperparameters,
            torch_dataset_preparation=True,
        )

    if "torch_dataset_preparation_backtest" in general_hyperparameters["stages"]:
        executor = Executor(
            experiment_id,
            general_hyperparameters,
            model_hyperparameters,
            torch_dataset_preparation=False,
            torch_dataset_preparation_backtest=True,
        )

    # For the 'training' and 'evaluation' stages, instantiate the executor with proper arguments.
    if (
        "training" in general_hyperparameters["stages"]
        or "evaluation" in general_hyperparameters["stages"]
    ):
        executor = Executor(
            experiment_id, general_hyperparameters, model_hyperparameters
        )

    if "training" in general_hyperparameters["stages"]:
        try:
            # Keep track of the files used in the training, validation and test sets.
            save_dataset_info(
                experiment_id=experiment_id,
                general_hyperparameters=general_hyperparameters,
            )
            # Train the model.
            executor.execute_training()
            # Clean up the experiment folder from wandb logging files.
            executor.logger_clean_up()
        except:
            print("Exception detected")
            wb_error_detection = True

    if (
        "evaluation" in general_hyperparameters["stages"]
        and wb_error_detection is False
    ):
        # Out-of-sample test of the model.
        executor.execute_testing()
        # Clean up the experiment folder from wandb logging files.
        executor.logger_clean_up()

    if "backtest" in general_hyperparameters["stages"]:
        # Backtest the model.
        market_sim.backtest(
            experiment_id=experiment_id, trading_hyperparameters=trading_hyperparameters
        )

    if "post_trading_analysis" in general_hyperparameters["stages"]:
        # Perform a post-trading analysis.
        post_trading_analysis.post_trading_analysis(
            experiment_id=experiment_id,
            general_hyperparameters=general_hyperparameters,
            trading_hyperparameters=trading_hyperparameters,
            model_hyperparameters=model_hyperparameters,
        )
