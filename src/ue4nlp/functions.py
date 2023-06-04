import numpy as np

def sep_features_by_class(X_features, scores):
    sep_features = {}
    for label in np.sort(np.unique(scores)):
        sep_features['{}'.format(label)] = X_features[scores == label]
    return sep_features

def diffclass_euclid_dist(target_feature, target_label, train_features_labels):
    min_dist = None
    for k, v in train_features_labels.items():
        if int(k) != target_label:
            for diff_vec in v:
                dist = np.linalg.norm(diff_vec-target_feature)
                if(min_dist is None or dist < min_dist):
                    min_dist = dist
    return min_dist

def sameclass_euclid_dist(target_feature, target_label, train_features_labels):
    min_dist = None
    for k, v in train_features_labels.items():
        if int(k) == target_label:
            for diff_vec in v:
                dist = np.linalg.norm(diff_vec-target_feature)
                if(min_dist is None or dist < min_dist):
                    min_dist = dist
    return min_dist