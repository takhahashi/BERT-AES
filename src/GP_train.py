import os
import hydra
import torch
import gpytorch
import numpy as np
from omegaconf import DictConfig
from transformers import AutoTokenizer
from utils.dataset import get_Dataset, get_score_range
from utils.cfunctions import simple_collate_fn
from utils.utils_models import create_module
from models.functions import extract_clsvec_truelabels
from models.models import GPModel


@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/1.AES/ASAP/BERT-AES/configs", config_name="GP_train")
def main(cfg: DictConfig):
    low, high = get_score_range(cfg.aes.prompt_id)
    train_dataset = get_Dataset('class', 
                                cfg.path.traindata_file_name, 
                                cfg.aes.prompt_id, 
                                AutoTokenizer.from_pretrained(cfg.scoring_model.model_name_or_path),
                                )

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=8,
                                                  shuffle=False,
                                                  collate_fn=simple_collate_fn,
                                                  )

    num_labels = high - low + 1
    if cfg.scoring_model.spectral_norm == True:
       print('SpectralNorm is applyed!')
       scoring_model_path = cfg.path.scoring_model_savepath + '_sepctralnorm'
       gp_save_path = cfg.path.save_path + '_spectralnorm'
    else:
       print('SpectralNorm is not applyed!')
       scoring_model_path = cfg.path.scoring_model_savepath
       gp_save_path = cfg.path.save_path
       
    classifier = create_module(model_name_or_path=cfg.scoring_model.model_name_or_path,
                               reg_or_class=cfg.scoring_model.reg_or_class,
                                learning_rate=1e-5,
                                num_labels=num_labels,
                                spectral_norm=cfg.scoring_model.spectral_norm
                                )
    classifier = classifier.cuda()
    classifier.load_state_dict(torch.load(scoring_model_path), strict=False)
    classifier.eval()

    word_vec, labels = extract_clsvec_truelabels(classifier, train_dataloader)
    train_x = torch.FloatTensor(word_vec)
    train_y = torch.FloatTensor(labels)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(train_x, train_y, likelihood)
    training_iter = cfg.training.iter_num
    model.train()
    likelihood.train()
    model.covar_module.base_kernel.lengthscale = np.linalg.norm(train_x[0].numpy() - train_x[1].numpy().T) ** 2 / 2
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=cfg.training.lr)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
      optimizer.zero_grad()
      output = model(train_x)
      loss = -mll(output, train_y)
      loss.backward()
      """
      print('Iter %d/%d - Loss: %.3f lengthscale: %.3f noise: %.3f' % (
          i+1, training_iter, loss.item(),
          model.covar_module.base_kernel.lengthscale.item(),
          model.likelihood.noise.item()
      ))
      """
      
      optimizer.step()

    torch.save(model.state_dict(), gp_save_path)


if __name__ == "__main__":
    main()