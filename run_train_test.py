import argparse
import yaml
import wandb
from train_test_autoencoder import run_train_test_model


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config_file', type=str, help='Path of the config file to use')
    parser.add_argument('--id_train_wandb', type=str, help='Path of the config file to use')
    opt = parser.parse_args()

    # 1 - load config file
    path_config_file = opt.path_config_file
    id_train_wandb = opt.id_train_wandb
    print("path_config_file: {}".format(path_config_file))
    cfg = load_config(path_config_file)


    if id_train_wandb is None:
        wandb.init(
            # set the wandb project where this run will be logged
            project="action_video_autoencoder",
            # track hyperparameters and run metadata
            config=cfg
        )
    else:
        wandb.init(
            # set the wandb project where this run will be logged
            project="action_video_autoencoder",
            # track hyperparameters and run metadata
            config=cfg,
            id=id_train_wandb,
            resume='must'
        )

    print("wandb.run.id = ", wandb.run.id)

    # 2 - run train and test
    do_train = cfg["model"].get("do_train", 1.0) > 0.0
    do_test = cfg["model"].get("do_test", 1.0) > 0.0
    run_train_test_model(cfg=cfg,
                         do_train=do_train,
                         do_test=do_test)
