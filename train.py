import hydra
from omegaconf import DictConfig, OmegaConf

# Registering a resolver that can return a callabale method
# This also works on classes, but you can also register get_class which is almost identical.
OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)

@hydra.main(version_base="1.1", config_path="configs/", config_name="train.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from training import train

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
