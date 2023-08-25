import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt


def down_sample(data, samples=300):
    new_data = [[], []]
    n = len(data[0])
    per_sample = n//samples
    for i in range(n):
        if (i%per_sample == 0) or (i+1 == n):
            new_data[0].append(data[0][i])
            new_data[1].append(data[1][i])
    return new_data

@hydra.main(config_path="/content/drive/MyDrive/GoogleColab/1.AES/ASAP/BERT-AES/configs", config_name="eval_ue_config")
def main(cfg: DictConfig):
    #utypes = ['simplevar', 'reg_dp', 'reg_mul', 'reg_trust_score', 'MP', 'class_dp_MP', 'class_dp_entropy', 'class_dp_epistemic', 'class_mul_MP', 'class_mul_entropy', 'class_mul_epistemic', 'class_trust_score', 'mix', 'mix_dp', 'mix_dp_entropy', 'mix_mul', 'mix_mul_entropy']
    utypes = ['simplevar', 'reg_mul', 'MP', 'class_trust_score', 'class_mul_MP', 'mix', 'mix_mul']
    ###roc_auc###
    roc_dic = {}
    for utype in utypes:
        roc_dic[utype] = []
    for prompt_id in range(1, 9):
        for utype in utypes:
            with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/pt{}/{}'.format(prompt_id, utype)) as f:
                fold_results = json.load(f)
            results = {k: np.array(v) for k, v in fold_results.items()}
            roc_dic[utype] = np.append(roc_dic[utype], np.round(results['roc'], decimals=3))
    for k, v in roc_dic.items():
        n_v = np.append(v, np.round(np.mean(v), decimals=3))
        roc_dic[k] = n_v
    roc_table = pd.DataFrame.from_dict(roc_dic, orient='index', columns=['pt1', 'pt2', 'pt3', 'pt4', 'pt5', 'pt6', 'pt7', 'pt8', 'mean'])
    roc_table.to_csv('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/roc_table.tsv', sep='\t', index=True)

    ##rpp##
    rpp_dic = {}
    for utype in utypes:
        rpp_dic[utype] = []
    for prompt_id in range(1, 9):
        for utype in utypes:
            with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/pt{}/{}'.format(prompt_id, utype)) as f:
                fold_results = json.load(f)
            results = {k: np.array(v) for k, v in fold_results.items()}
            rpp_dic[utype] = np.append(rpp_dic[utype], np.round(results['rpp'], decimals=3))
    for k, v in rpp_dic.items():
        n_v = np.append(v, np.round(np.mean(v), decimals=3))
        rpp_dic[k] = n_v
    rpp_table = pd.DataFrame.from_dict(rpp_dic, orient='index', columns=['pt1', 'pt2', 'pt3', 'pt4', 'pt5', 'pt6', 'pt7', 'pt8', 'mean'])
    rpp_table.to_csv('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/rpp_table.tsv', sep='\t', index=True)

    ##rcc###
    rcc_dic = {}
    for utype in utypes:
        rcc_dic[utype] = []
    for prompt_id in range(1, 9):
        for utype in utypes:
            with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/pt{}/{}'.format(prompt_id, utype)) as f:
                fold_results = json.load(f)
            results = {k: np.array(v) for k, v in fold_results.items()}
            rcc_dic[utype] = np.append(rcc_dic[utype], np.round(results['rcc'], decimals=3))
    for k, v in rcc_dic.items():
        n_v = np.append(v, np.round(np.mean(v), decimals=3))
        rcc_dic[k] = n_v
    rcc_table = pd.DataFrame.from_dict(rcc_dic, orient='index', columns=['pt1', 'pt2', 'pt3', 'pt4', 'pt5', 'pt6', 'pt7', 'pt8', 'mean'])
    rcc_table.to_csv('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/rcc_table/{}_table.tsv'.format(cfg.rcc.metric_type), sep='\t', index=True)

    ##rcc_y_fig###
    for prompt_id in range(1, 9):
        plt.figure()
        for utype in utypes:
            with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/pt{}/{}'.format(prompt_id, utype)) as f:
                fold_results = json.load(f)
            results = {k: np.array(v) for k, v in fold_results.items()}
            mean_rcc_y = results['rcc_y']
            fraction = 1 / len(mean_rcc_y)
            rcc_x = [fraction]
            for i in range(len(mean_rcc_y)-1):
                rcc_x = np.append(rcc_x, fraction+rcc_x[-1])
            down_data = down_sample([rcc_x, mean_rcc_y], samples=50)
            plt.plot(down_data[0], down_data[1], label=utype)
        plt.legend()
        plt.savefig('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/rcc_fig/{}_pt{}.png'.format(cfg.rcc.metric_type, prompt_id)) 
        plt.show()

    #table_idx_name = ['simple_reg', 'dp_reg', 'mul_reg', 'simple_class', 'dp_class', 'mul_class', 'mix', 'dp_mix', 'mul_mix']
    #utype_path_name = ['simple_reg_acc', 'dp_reg_acc', 'ense_reg_acc', 'simple_class_acc', 'dp_class_acc', 'ense_class_acc', 'mix_acc', 'dp_mix_acc', 'ense_mix_acc']
    table_idx_name = ['reg', 'mul_reg', 'class', 'class_reg', 'mix', 'mul_mix']
    utype_path_name = ['simple_reg_acc', 'ense_reg_acc', 'simple_class_acc', 'ense_class_acc', 'mix_acc', 'ense_mix_acc']

    qwk_dic = {}
    for utype in table_idx_name:
        qwk_dic[utype] = []
    for prompt_id in range(1, 9):
        for idx, utype in enumerate(utype_path_name):
            with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/pt{}/{}'.format(prompt_id, utype)) as f:
                fold_results = json.load(f)
            results = {k: np.array(v) for k, v in fold_results.items()}
            qwk_dic[table_idx_name[idx]] = np.append(qwk_dic[table_idx_name[idx]], np.round(results['qwk'], decimals=3))
    for k, v in qwk_dic.items():
        n_v = np.append(v, np.round(np.mean(v), decimals=3))
        qwk_dic[k] = n_v
    qwk_table = pd.DataFrame.from_dict(qwk_dic, orient='index', columns=['pt1', 'pt2', 'pt3', 'pt4', 'pt5', 'pt6', 'pt7', 'pt8', 'mean'])
    qwk_table.to_csv('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/qwk_table.tsv', sep='\t', index=True)

    corr_dic = {}
    for utype in table_idx_name:
        corr_dic[utype] = []
    for prompt_id in range(1, 9):
        for idx, utype in enumerate(utype_path_name):
            with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/pt{}/{}'.format(prompt_id, utype)) as f:
                fold_results = json.load(f)
            results = {k: np.array(v) for k, v in fold_results.items()}
            corr_dic[table_idx_name[idx]] = np.append(corr_dic[table_idx_name[idx]], np.round(results['corr'], decimals=3))
    for k, v in corr_dic.items():
        n_v = np.append(v, np.round(np.mean(v), decimals=3))
        corr_dic[k] = n_v
    corr_table = pd.DataFrame.from_dict(corr_dic, orient='index', columns=['pt1', 'pt2', 'pt3', 'pt4', 'pt5', 'pt6', 'pt7', 'pt8', 'mean'])
    corr_table.to_csv('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/corr_table.tsv', sep='\t', index=True)

    rmse_dic = {}
    for utype in table_idx_name:
        rmse_dic[utype] = []
    for prompt_id in range(1, 9):
        for idx, utype in enumerate(utype_path_name):
            with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/pt{}/{}'.format(prompt_id, utype)) as f:
                fold_results = json.load(f)
            results = {k: np.array(v) for k, v in fold_results.items()}
            rmse_dic[table_idx_name[idx]] = np.append(rmse_dic[table_idx_name[idx]], np.round(results['rmse'], decimals=3))
    for k, v in rmse_dic.items():
        n_v = np.append(v, np.round(np.mean(v), decimals=3))
        rmse_dic[k] = n_v
    rmse_table = pd.DataFrame.from_dict(rmse_dic, orient='index', columns=['pt1', 'pt2', 'pt3', 'pt4', 'pt5', 'pt6', 'pt7', 'pt8', 'mean'])
    rmse_table.to_csv('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/rmse_table.tsv', sep='\t', index=True)

if __name__ == "__main__":
    main()