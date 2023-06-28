import numpy as np
import pandas as pd
import json
def main():
    ###roc_auc###
    roc_dic = {}
    for utype in ['simplevar', 'reg_dp', 'reg_mul', 'reg_trust_score', 'MP', 'class_dp', 'class_mul', 'class_trust_score']:
        roc_dic[utype] = []
    for prompt_id in range(1, 9):
        for utype in ['simplevar', 'reg_dp', 'reg_mul', 'reg_trust_score', 'MP', 'class_dp', 'class_mul', 'class_trust_score']:
            with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/pt{}/{}'.format(prompt_id, utype)) as f:
                fold_results = json.load(f)
            results = {k: np.array(v) for k, v in fold_results.items()}
            roc_dic[utype] = np.append(roc_dic[utype], np.round(results['roc'], decimals=3))
    for k, v in roc_dic.items():
        n_v = np.append(v, np.round(np.mean(v), decimals=3))
        roc_dic[k] = n_v
    roc_table = pd.DataFrame.from_dict(roc_dic, orient='index', columns=['pt1', 'pt2', 'pt3', 'pt4', 'pt5', 'pt6', 'pt7', 'pt8', 'mean'])
    roc_table.to_csv('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/roc_talbe.tsv', sep='\t', index=True)

    ##rpp##
    rpp_dic = {}
    for utype in ['simplevar', 'reg_dp', 'reg_mul', 'reg_trust_score', 'MP', 'class_dp', 'class_mul', 'class_trust_score']:
        rpp_dic[utype] = []
    for prompt_id in range(1, 9):
        for utype in ['simplevar', 'reg_dp', 'reg_mul', 'reg_trust_score', 'MP', 'class_dp', 'class_mul', 'class_trust_score']:
            with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/pt{}/{}'.format(prompt_id, utype)) as f:
                fold_results = json.load(f)
            results = {k: np.array(v) for k, v in fold_results.items()}
            rpp_dic[utype] = np.append(rpp_dic[utype], np.round(results['rpp'], decimals=3))
    for k, v in rpp_dic.items():
        n_v = np.append(v, np.round(np.mean(v), decimals=3))
        rpp_dic[k] = n_v
    rpp_table = pd.DataFrame.from_dict(rpp_dic, orient='index', columns=['pt1', 'pt2', 'pt3', 'pt4', 'pt5', 'pt6', 'pt7', 'pt8', 'mean'])
    rpp_table.to_csv('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/rpp_talbe.tsv', sep='\t', index=True)

    ##rcc###
    rcc_dic = {}
    for utype in ['simplevar', 'reg_dp', 'reg_mul', 'reg_trust_score', 'MP', 'class_dp', 'class_mul', 'class_trust_score']:
        rcc_dic[utype] = []
    for prompt_id in range(1, 9):
        for utype in ['simplevar', 'reg_dp', 'reg_mul', 'reg_trust_score', 'MP', 'class_dp', 'class_mul', 'class_trust_score']:
            with open('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/pt{}/{}'.format(prompt_id, utype)) as f:
                fold_results = json.load(f)
            results = {k: np.array(v) for k, v in fold_results.items()}
            rcc_dic[utype] = np.append(rcc_dic[utype], np.round(results['rcc'], decimals=3))
    for k, v in rcc_dic.items():
        n_v = np.append(v, np.round(np.mean(v), decimals=3))
        rcc_dic[k] = n_v
    rcc_table = pd.DataFrame.from_dict(rcc_dic, orient='index', columns=['pt1', 'pt2', 'pt3', 'pt4', 'pt5', 'pt6', 'pt7', 'pt8', 'mean'])
    rcc_table.to_csv('/content/drive/MyDrive/GoogleColab/1.AES/ASAP/torchlightning/rcc_talbe.tsv', sep='\t', index=True)

    

if __name__ == "__main__":
    main()