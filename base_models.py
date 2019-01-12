from copy import copy
from xgboost.sklearn import XGBClassifier
from sklearn.decomposition import FastICA
from sklearn.ensemble import ExtraTreesClassifier

def get_model():
    return XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=7,
        min_child_weight=2,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=8,
        scale_pos_weight=1,
        seed=27
    )

def get_base_model():
    return make_pipeline(
        make_union(
            FunctionTransformer(copy),
            FastICA(tol=0.05)
        ),
        ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.6500000000000001,
                             min_samples_leaf=1, min_samples_split=5, n_estimators=100)
    )


def get_XGBClassifier():
    model=XGBClassifier()
    return model
