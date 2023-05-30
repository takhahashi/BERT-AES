from omegaconf import DictConfig, OmegaConf
import hydra
 
@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/1.AES/ASAP/test1/configs", config_name="config")
def my_app(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
     # raises an exception
 
if __name__ == "__main__":
    my_app()