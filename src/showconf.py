from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/1.AES/ASAP/test1/configs", config_name="config")
def show(cfg):
    x = cfg.path.checkpoint_path
    print(f"type: {type(x).__name__}, value: {repr(x)}")