# Initialize the models here + train/validate them
from torch import nn
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import f1_score, silhouette_score, accuracy_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from skmultilearn.model_selection.iterative_stratification import IterativeStratification
import math
import random
from copy import deepcopy
import itertools
from config.config import dataset_configs

def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.enabled = False

def get_ml_model(model_params, n_classes, target_type,):
    if model_params["model_specific"] == "xgb":
        if target_type == "reg":
            # OG
            return XGBRegressor(n_estimators=int(model_params["n_estimators"]),
                                max_depth=int(model_params["max_depth"]),
                                learning_rate=model_params["learning_rate"],
                                subsample=model_params["subsample"],
                            colsample_bytree=model_params["colsample_bytree"],
                            random_state=model_params["seed"],)
        else:
            if n_classes > 2:
                return XGBClassifier(n_estimators=int(model_params["n_estimators"]),
                                    max_depth=int(model_params["max_depth"]),
                                    learning_rate=model_params["learning_rate"],
                                    subsample=model_params["subsample"],
                                    colsample_bytree=model_params["colsample_bytree"],
                                    random_state=model_params["seed"],
                                    num_class=n_classes,)
            else:
                return XGBClassifier(n_estimators=int(model_params["n_estimators"]),
                                    max_depth=int(model_params["max_depth"]),
                                    learning_rate=model_params["learning_rate"],
                                    subsample=model_params["subsample"],
                                    colsample_bytree=model_params["colsample_bytree"],
                                    random_state=model_params["seed"],)
    elif model_params["model_specific"] == "rf":
        if target_type == "reg":
            return RandomForestRegressor(
                n_estimators=int(model_params["n_estimators"]),
                max_depth=int(model_params["max_depth"]),
                min_samples_leaf=int(model_params["min_samples_leaf"]),
                random_state=model_params["seed"],
            )
    elif model_params["model_specific"] == "ridge":
        if target_type == "reg":
            return Ridge(
                alpha=model_params["alpha"],
                fit_intercept=model_params["fit_intercept"],
                solver=model_params["solver"],
                random_state=model_params["seed"]
            )
    elif model_params["model_specific"] == "elastic":
        if target_type == "reg":
            return ElasticNet(
                alpha=model_params["alpha_elastic"],
                l1_ratio=model_params["l1_ratio_elastic"],
                fit_intercept=model_params["fit_intercept_elastic"],
                max_iter=5000,
                random_state=model_params["seed"]
            )
    elif model_params["model_specific"] == "linear":
        if target_type == "reg":
            return LinearRegression()
    elif model_params["model_specific"] == "svr":
        if target_type == "reg":
            return SVR(
                C=model_params["C"],
                epsilon=model_params["epsilon"],
                kernel=model_params["kernel"],
                degree=int(model_params["degree"]),
                gamma=model_params["gamma"]
            )

class XGBRegressor(XGBRegressor):
    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)

class RandomForestRegressor(RandomForestRegressor):
    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)

class XGBClassifier(XGBClassifier):
    def set_threshold(self, threshold):
        self.threshold = threshold

class Ridge(Ridge):
    def score(self, X, y, sample_weight=None):
        """
        Override .score() to return MAE instead of default R²
        """
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)

class ElasticNet(ElasticNet):
    def score(self, X, y, sample_weight=None):
        """
        Override .score() to return MAE instead of default R²
        """
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)

class SVR(SVR):
    def score(self, X, y, sample_weight=None):
        """
        Override .score() to return MAE instead of default R²
        """
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred, sample_weight=sample_weight)

class MLCausalRegressor():
    def __init__(self, model_params):
        self.model_params = model_params
        self.single_model = True if ("S" in model_params["method"] and self.model_params["target"] == "outcome") else False
        self.target = model_params["target"]
        self.n_classes = model_params["dim_t"] if model_params["dim_t"] > 1 else 2
        model = get_ml_model(model_params=model_params, n_classes=self.n_classes, target_type="reg")
        self.model = [deepcopy(model) for _ in range(self.n_classes)] if not self.single_model else deepcopy(model)

    def get_ind(self, T, t):
        if self.n_classes == 2:
            return np.where(T == t, 1, 0).astype(float).flatten()
        else:
            return (T.sum(axis=1) == 0).astype(float) if t == 0 else T[:, t-1].astype(float)

    def prep(self, X, T, Y):
        if isinstance(T, torch.Tensor):
            if self.n_classes > 2:
                T = torch.argmax(T, dim=1).to(torch.int)
            else:
                T = T.to(torch.int)
            T = T.cpu().numpy()  # convert to NumPy for downstream use
        else:  # Assume it's a NumPy array
            if self.n_classes > 2:
                T = np.argmax(T, axis=1).astype(int)
            else:
                T = T.astype(int)
        
        if len(np.unique(T)) == self.n_classes:
            transformer = OneHotEncoder(sparse_output=False, drop='first')
            T = transformer.fit_transform(T.reshape(-1, 1))
            if Y.ndim == 2 and Y.shape[1] == 1:
                Y = Y.flatten()
        # check whether Y has multiple dimensions and switch dimensions if so
        if Y.ndim == 2 and Y.shape[1] > Y.shape[0]:
            Y = Y.T

        if T.ndim == 1:
            T = np.array(T)
            T = T.reshape(-1, 1)

        return X, T, Y

    def fit_and_get_loss(self, x_case, x_event, t, prefix_len, y, weights, tuning=False, dataset_ps=None):
        X, T, Y = self.prep(x_case, t, y)
        rng = np.random.default_rng(seed=self.model_params["seed"])

        if tuning:
            _, counts = np.unique(T, axis=0, return_counts=True)
            min_samples_per_treatment = counts.min()

            # each fold needs at least 2 samples per treatment
            max_possible_splits = min(5, min_samples_per_treatment // 2)

            if max_possible_splits >= 2:
                # use iterative stratified k-fold
                n_splits = max_possible_splits
                mskf = IterativeStratification(n_splits=n_splits, order=1)
                split_generator = mskf.split(X, T)
            else:
                # fallback: single stratified train/test split
                train_idx, test_idx = train_test_split(
                    np.arange(len(X)),
                    test_size=0.3 if len(X) > 1 else 0.5,  # avoid degenerate split
                    random_state=42,
                    stratify=T if len(np.unique(T)) > 1 else None,  # only stratify if possible
                )
                split_generator = [(train_idx, test_idx)]

            # kf = KFold(n_splits=5, shuffle=True, random_state=self.model_params["seed"])

            losses = []
            
            # Convert each row in T into a tuple to represent a unique treatment level
            # T_levels = np.array([tuple(row) for row in T])
            T_levels = np.array([tuple(row) if hasattr(row, '__iter__') else (row,) for row in T])
            unique_levels = np.unique(T_levels, axis=0)

            for train_idx, test_idx in split_generator:
                train_levels = np.unique(T_levels[train_idx], axis=0)
                test_levels = np.unique(T_levels[test_idx], axis=0)
                # Find which treatment levels are missing in test
                missing_in_test = [
                    lvl for lvl in unique_levels
                    if not any(np.all(lvl == tl) for tl in test_levels)
                ]
                if missing_in_test:
                    for lvl in missing_in_test:
                        # find all indices in train with this treatment level
                        lvl_indices_in_train = np.where(np.all(T_levels[train_idx] == lvl, axis=1))[0]
                        if len(lvl_indices_in_train) > 0:
                            n_to_move = max(1, len(lvl_indices_in_train) // 4)  # move 1/4th (at least one)
                            move_rel_idxs = rng.choice(lvl_indices_in_train, size=n_to_move, replace=False)
                            move_abs_idxs = train_idx[move_rel_idxs]
                            # move from train → test
                            train_idx = np.setdiff1d(train_idx, move_abs_idxs)
                            test_idx = np.concatenate([test_idx, move_abs_idxs])
                # (optional) check correctness
                test_levels_after = np.unique(T_levels[test_idx], axis=0)
                assert all(
                    any(np.all(lvl == tl) for tl in test_levels_after)
                    for lvl in unique_levels
                ), "Some treatment levels missing in test!"

                X_train, X_test = X[train_idx], X[test_idx]
                T_train, T_test = T[train_idx], T[test_idx]
                Y_train, Y_test = Y[train_idx], Y[test_idx]

                fold_model = get_ml_model(model_params=self.model_params, n_classes=self.n_classes, target_type="reg")
                model = [deepcopy(fold_model) for _ in range(self.n_classes)] if not self.single_model else deepcopy(fold_model)

                if self.single_model:
                    feat = np.concatenate((X_train, T_train), axis=1) if self.target == "outcome" else X_train
                    model.fit(feat, Y_train)
                    loss = self.score(model=model, X=X_test, T=T_test, Y=Y_test)
                else:
                    loss = 0
                    for t in range(self.n_classes):
                        ind_t_train = self.get_ind(T_train, t)
                        Y_to_fit_to = Y_train[ind_t_train == 1] if self.target == "outcome" else Y_train[:, t]
                        X_to_fit_to = X_train[ind_t_train == 1] if self.target == "outcome" else X_train
                        model[t].fit(X_to_fit_to, Y_to_fit_to)
                        
                        ind_t_test = self.get_ind(T_test, t)
                        Y_test_to_fit_to = Y_test[ind_t_test == 1] if self.target == "outcome" else Y_test[:, t]
                        X_test_to_fit_to = X_test[ind_t_test == 1] if self.target == "outcome" else X_test
                        loss += self.score(model=model[t], X=X_test_to_fit_to, T=T_test[ind_t_test == 1], Y=Y_test_to_fit_to)
                    loss = loss / self.n_classes

                losses.append(loss)

            return np.mean(losses)

        else:
            if self.single_model:
                feat = np.concatenate((X, T), axis=1) if self.target == "outcome" else X
                self.model.fit(feat, Y)
                loss = self.score(model=self.model, X=X, T=T, Y=Y)
            else:
                loss = 0
                for t in range(self.n_classes):
                    ind_t = self.get_ind(T, t)
                    Y_to_fit_to = Y[ind_t == 1] if self.target == "outcome" else Y[:, t]
                    X_to_fit_to = X[ind_t == 1] if self.target == "outcome" else X
                    self.model[t].fit(X_to_fit_to, Y_to_fit_to)
                    loss += self.score(model=self.model[t], X=X_to_fit_to, T=T[ind_t == 1], Y=Y_to_fit_to)
                loss = loss / self.n_classes
            return loss

    def forward(self, x_case, x_event, t, prefix_len, y, ret_counterfactuals=False):
        X, _, _ = self.prep(x_case, t, y)

        # if self.single_model and self.target == "effect":
        #     return self.model.predict(X).T

        preds = np.zeros((self.n_classes, X.shape[0])) 
        for treatment in range(self.n_classes):
            if self.single_model and self.target == "outcome":
                T_artificial = np.zeros((len(X), 1)) if self.n_classes == 2 else np.zeros((len(X), self.n_classes - 1))
                if self.n_classes > 2:
                    if treatment != 0:
                        T_artificial[:, treatment - 1] = 1
                else:
                    T_artificial[:] = treatment
                feat = np.concatenate((X, T_artificial), axis=1) if self.target == "outcome" else X
                preds[treatment, :] = self.model.predict(feat)
            else:
                preds[treatment, :] = self.model[treatment].predict(X)
        return preds

    def score(self, model, X, T, Y):
        if self.single_model:
            feat = np.concatenate((X, T), axis=1) if self.target == "outcome" else X
            return model.score(X=feat, y=Y)
        else:
            return model.score(X=X, y=Y)


class XGB_Opt_Class():
    def __init__(self, model_params):
        self.model_params = model_params
        self.n_classes = model_params["dim_t"] if model_params["dim_t"] > 1 else 2  # If binary treatment, we use one-hot encoding, so dim_t = 1

        if self.n_classes > 2:
            # If multi-class treatment, we use XGBClassifier
            self.model = XGBClassifier(n_estimators=int(model_params["n_estimators"]),
                                        max_depth=int(model_params["max_depth"]),
                                        learning_rate=model_params["learning_rate"],
                                        subsample=model_params["subsample"],
                                        colsample_bytree=model_params["colsample_bytree"],
                                        random_state=model_params["seed"],
                                        num_class=self.n_classes,)
        else:
            self.model = XGBClassifier(n_estimators=int(model_params["n_estimators"]),
                                        max_depth=int(model_params["max_depth"]),
                                        learning_rate=model_params["learning_rate"],
                                        subsample=model_params["subsample"],
                                        colsample_bytree=model_params["colsample_bytree"],
                                        random_state=model_params["seed"],)
    
    def forward(self, x_case, x_event, t, prefix_len, y, ret_counterfactuals=False):
        # x_case is the input for the model, x_event is not used here
        # t is the treatment, prefix_len is the length of the prefix
        # we return the action/treatment that is most likely to be optimal
        proba = self.model.predict_proba(x_case)
        if self.model.threshold is not None:
            if self.n_classes == 2:
                # Binary thresholding
                y_pred_thresh = (proba[:, 1] >= self.model.threshold).astype(int)
                
                y_pred_onehot = np.zeros((len(y_pred_thresh), 2), dtype=int)
                y_pred_onehot[np.arange(len(y_pred_thresh)), y_pred_thresh] = 1
                return y_pred_onehot    
            else:
                # Multiclass thresholding
                preds = []
                for prob in proba:
                    # candidates: all classes above their threshold
                    candidates = [k for k, p in enumerate(prob) if p >= self.model.threshold[k]]
                    if len(candidates) == 0:
                        pred = np.argmax(prob)
                    else:
                        pred = max(candidates, key=lambda k: prob[k])
                    preds.append(pred)
                n_classes = proba.shape[1]
                preds_onehot = np.zeros((len(preds), n_classes), dtype=int)
                preds_onehot[np.arange(len(preds)), preds] = 1
                return preds_onehot
        else:
            return proba
    
    def expand_data(self, x_case, weights):
        n_samples, _ = x_case.shape
        X_exp = np.repeat(x_case, self.n_classes, axis=0)
        y_exp = np.tile(np.arange(self.n_classes), n_samples)
        w_exp = weights.reshape(-1)
        return X_exp, y_exp, w_exp
   
    def fit_and_get_loss(self, x_case, x_event, t, prefix_len, y, weights, tuning=False, dataset_ps=None):
        # NOTE: here the y is the optimal estimated treatment, not the outcome
        input = x_case

        if tuning:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.model_params["seed"])

            average = 'macro' if self.n_classes > 2 else 'binary'

            losses = []

            for train_idx, test_idx in skf.split(input, y):
                X_train, X_test = input[train_idx], input[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                w_train = weights[train_idx]

                maxs = np.max(w_train, axis=1, keepdims=True)  # per sample max over classes
                w_training_weights = np.abs(w_train - maxs)
                # use min max normalization to keep weights in [0, 1]
                if np.max(w_training_weights) > 0:
                    w_training_weights = (w_training_weights - np.min(w_training_weights)) / (np.max(w_training_weights) - np.min(w_training_weights))
                X_train_repeat, y_train_repeat, w_training_weights = self.expand_data(X_train, w_training_weights)

                # # just make all the weights equal to 1
                # w_training_weights = np.ones_like(y_train)
                # X_train_repeat, y_train_repeat, w_training_weights = X_train, y_train, w_training_weights

                # check if all classes are present in y
                if len(np.unique(y_train)) < self.n_classes:
                    # add a dummy
                    dummy_class = np.setdiff1d(np.arange(self.n_classes), np.unique(y_train))
                    y_train_repeat = np.concatenate((y_train_repeat, dummy_class))
                    w_training_weights = np.concatenate((w_training_weights, np.zeros_like(w_training_weights[:len(dummy_class)])))
                    X_train_repeat = np.concatenate((X_train_repeat, np.zeros((len(dummy_class), X_train_repeat.shape[1]))), axis=0)
                
                if self.n_classes > 2:
                    # If multi-class treatment, we use XGBClassifier
                    model = XGBClassifier(
                        n_estimators=int(self.model_params["n_estimators"]),
                        max_depth=int(self.model_params["max_depth"]),
                        learning_rate=self.model_params["learning_rate"],
                        subsample=self.model_params["subsample"],
                        colsample_bytree=self.model_params["colsample_bytree"],
                        random_state=self.model_params["seed"],
                        num_class=self.n_classes,
                    )
                else:
                    model = XGBClassifier(
                        n_estimators=int(self.model_params["n_estimators"]),
                        max_depth=int(self.model_params["max_depth"]),
                        learning_rate=self.model_params["learning_rate"],
                        subsample=self.model_params["subsample"],
                        colsample_bytree=self.model_params["colsample_bytree"],
                        random_state=self.model_params["seed"],
                    )

                model.fit(X_train_repeat, y_train_repeat, sample_weight=w_training_weights)

                # NEW: instance-dependent cost-sensitive thresholding
                y_prob = model.predict_proba(X_test)
                if self.n_classes == 2:
                    # Binary threshold search
                    thresholds = np.linspace(0.1, 0.9, 9)
                    best_loss, best_thresh = float("inf"), 0.5
                    for tau in thresholds:
                        y_pred_thresh = (y_prob[:, 1] >= tau).astype(int)
                        loss = self.custom_loss_function(y_test, y_pred_thresh, weights, test_idx)
                        if loss < best_loss:
                            best_loss, best_thresh = loss, tau
                    score = best_loss

                else:
                    # Multiclass threshold search
                    thresholds = np.linspace(0.1, 0.9, 9)
                    best_loss, best_thresh = float("inf"), [0.5] * self.n_classes

                    # Try grid search (can get expensive if n_classes large)
                    for tau_vec in itertools.product(thresholds, repeat=self.n_classes):
                        preds = []
                        for prob in y_prob:
                            # candidates: all classes above their threshold
                            candidates = [k for k, p in enumerate(prob) if p >= tau_vec[k]]
                            if len(candidates) == 0:
                                pred = np.argmax(prob)  # fallback
                            else:
                                pred = max(candidates, key=lambda k: prob[k])
                            preds.append(pred)
                        preds = np.array(preds)

                        loss = self.custom_loss_function(y_test, preds, weights, test_idx)
                        if loss < best_loss:
                            best_loss, best_thresh = loss, tau_vec

                    print("Best per-class thresholds:", best_thresh)
                    score = best_loss
                
                # Remember the best threshold for future use
                model.set_threshold(best_thresh)

                losses.append(score)
                
                # # OLD
                # y_pred = model.predict(X_test)
                # # score = - f1_score(y_test, y_pred, average=average)
                # # score = - accuracy_score(y_test, y_pred)  # Use accuracy for tuning
                # score = self.custom_loss_function(y_test, y_pred, weights, test_idx)
                # losses.append(score)

            return np.mean(losses)
        else:
            maxs = np.max(weights, axis=1, keepdims=True)  
            training_weights = np.abs(weights - maxs)
            # use min max normalization to keep weights in [0, 1]
            if np.max(training_weights) > 0:
                training_weights = (training_weights - np.min(training_weights)) / (np.max(training_weights) - np.min(training_weights))
            input_repeat, y_repeat, training_weights = self.expand_data(x_case, training_weights)

            # # just make all the weights equal to 1
            # training_weights = np.ones_like(y)
            # input_repeat, y_repeat, training_weights = x_case, y, training_weights

            # check if all classes are present in y
            if len(np.unique(y_repeat)) < self.n_classes:
                # add a dummy
                dummy_class = np.setdiff1d(np.arange(self.n_classes), np.unique(y_repeat))
                y_repeat = np.concatenate((y_repeat, dummy_class))
                training_weights = np.concatenate((training_weights, np.zeros_like(training_weights[:len(dummy_class)])))
                input_repeat = np.concatenate((input_repeat, np.zeros((len(dummy_class), input_repeat.shape[1]))), axis=0)

            self.model.fit(input_repeat, y_repeat, sample_weight=training_weights)
            
            # NEW: instance-dependent cost-sensitive thresholding
            # Predict probabilities
            y_prob = self.model.predict_proba(input)

            if self.n_classes == 2:
                # Binary threshold search
                thresholds = np.linspace(0.1, 0.9, 9)
                best_loss, best_thresh = float("inf"), 0.5
                for tau in thresholds:
                    preds = (y_prob[:, 1] >= tau).astype(int)
                    loss = self.custom_loss_function(y, preds, weights, np.arange(len(y)))
                    if loss < best_loss:
                        best_loss, best_thresh = loss, tau

                print("Best binary threshold:", best_thresh)
                preds = (y_prob[:, 1] >= best_thresh).astype(int)
            else:
                # Multi-class threshold search
                thresholds = np.linspace(0.1, 0.9, 9)
                best_loss, best_thresh = float("inf"), [0.5] * self.n_classes

                for tau_vec in itertools.product(thresholds, repeat=self.n_classes):
                    preds = []
                    for prob in y_prob:
                        candidates = [k for k, p in enumerate(prob) if p >= tau_vec[k]]
                        if len(candidates) == 0:
                            pred = np.argmax(prob)  # fallback
                        else:
                            pred = max(candidates, key=lambda k: prob[k])
                        preds.append(pred)
                    preds = np.array(preds)

                    loss = self.custom_loss_function(y, preds, weights, np.arange(len(y)))
                    if loss < best_loss:
                        best_loss, best_thresh = loss, tau_vec

                print("Best per-class thresholds:", best_thresh)
                preds = []
                for prob in y_prob:
                    candidates = [k for k, p in enumerate(prob) if p >= best_thresh[k]]
                    if len(candidates) == 0:
                        pred = np.argmax(prob)
                    else:
                        pred = max(candidates, key=lambda k: prob[k])
                    preds.append(pred)
                preds = np.array(preds)

            self.model.set_threshold(best_thresh)

            # Compute evaluation metrics for reference
            f1_score_value = f1_score(y, preds, average='macro' if self.n_classes > 2 else 'binary')

            return -f1_score_value  # minimize negative F1 as your loss

            # # OLD
            # # return f1-macro or f1 score
            # preds = self.model.predict(input)
            # acc_score_value = accuracy_score(y, preds)
            # if self.n_classes > 2:
            #     f1_score_value = f1_score(y, preds, average='macro')
            #     print("The F1 score is:", f1_score_value)
            #     print("The Accuracy score is:", acc_score_value)
            #     return - f1_score_value
            #     # return - accuracy_score(y, preds)  # Use accuracy for evaluation
            # else:
            #     f1_score_value = f1_score(y, preds)
            #     print("The F1 score is:", f1_score_value)
            #     print("The Accuracy score is:", acc_score_value)
            #     return - f1_score_value
            #     # return - accuracy_score(y, preds)
        
    def custom_loss_function(self, y_true, y_pred, penalties, sample_idx):
        loss = 0.0
        for i, yi, yp in zip(sample_idx, y_true, y_pred):
            if yi != yp:  # penalize only misclassifications (optional)
                loss += penalties[i, yp]
        return loss / len(y_true)
    
class KMeansClustering():
    def __init__(self, model_params):
        self.model_params = model_params
        self.model = KMeans(n_clusters=int(model_params["n_clusters"]),
                            init='k-means++',
                            n_init=int(model_params["n_init"]),
                            random_state=model_params["seed"])
        # print('K: ', model_params["n_clusters"])

        # IMPORTANT NOTE: in Branchi, the authors use the reward up until the current prefix also in their clustering; but here, we do not include the reward
        # The reward will always be 0 up until the current prefix

    def drop_irrelevant_columns(self, x_case):
        # delete any useless columns if they are present: case_nr, a, activity, outcome
        x_case = x_case.drop(columns=['case_nr', 'a', 'activity', 'outcome'], errors='ignore')

        # drop if not in model_params["feature_names"]
        selected_features = [col for col, keep in self.model_params["feature_names"].items() if keep == 1]

        if len(selected_features) == 0:
            selected_features = x_case.columns.tolist()  # if none selected, keep all

        x_case = x_case[selected_features]

        return x_case
        
    def fit_and_get_loss(self, x_case, x_event, t, prefix_len, y, weights=None, tuning=False, test_percent=0.2):
        x_case = self.drop_irrelevant_columns(x_case)
        # # TEST: drop every column with count/pos in it --> THIS HEAVILY LOWERED PERFORMANCE
        # x_case = x_case.drop(columns=[col for col in x_case.columns if ("_count" in col or "_pos" in col)])

        # Split into train/test
        X_train, X_test = train_test_split(x_case, test_size=test_percent, random_state=self.model_params["seed"]) if test_percent > 0 else (x_case, x_case)
        
        # Fit on training data
        self.model.fit(X_train)
        
        # Predict on test data
        test_labels = self.model.predict(X_test)
        
        # Calculate silhouette score (higher is better)
        silhouette_avg = silhouette_score(X_test, test_labels)
        
        # Return negative for minimization
        return -silhouette_avg

    def forward(self, x_case, x_event, t, prefix_len):
        x_case = self.drop_irrelevant_columns(x_case)

        # # TEST: drop every column with count/pos in it --> THIS HEAVILY LOWERED PERFORMANCE
        # x_case = x_case.drop(columns=[col for col in x_case.columns if ("_count" in col or "_pos" in col)] )
        
        # Predict the cluster labels for the input data
        cluster_labels = self.model.predict(x_case)
        return cluster_labels
    
class KMeans_QLearning():
    """
    Initialize the Q-learning model based on the provided model parameters.
    """
    def __init__(self, model_params):
        self.model_params = model_params
        self.random_object = random.Random(model_params["seed"])
        self.model = [None, None]  # Placeholder for the clustering model and the Q-table
        self.kmeans_clustering = None
        self.q_table_final = None

        # # load the KMeans clustering model
        # self.kmeans_clustering = KMeansClustering(model_params=model_params)
        # with open(model_params["savepath_cluster_model"] + ".pkl", 'rb') as f:
        #     self.kmeans_clustering.model = pickle.load(f)
    
    def fit_and_get_loss(self, x_case, x_event, t, prefix_len, y, weights=None, tuning=False):
        nr_episodes_train = int(self.model_params["train_size"] * 0.8) if tuning else self.model_params["train_size"]
        nr_episodes_eval = int(self.model_params["train_size"] * 0.2) if tuning else 0

        # # cluster the datas
        # silhouette = 0
        # self.model_params["n_clusters"] = 2
        # stop = False
        # counter = 0
        # best_silhouette = 1
        # while not stop:
        #     self.model_params["n_clusters"] += 1
        #     self.kmeans_clustering, silhouette = self.cluster_data(x_case)
        #     print('Silhouette: ', silhouette, ", N_clusters: ", self.model_params["n_clusters"])
        #     if silhouette < best_silhouette:
        #         best_silhouette = silhouette
        #         counter = 0
        #     else:
        #         counter += 1
        #     if counter > 20:
        #         stop = True
        # self.model_params["n_clusters"] = 19
        
        self.kmeans_clustering, silhouette = self.cluster_data(x_case)
        
        # construct the MDP
        mdp_df = self.construct_mdp(data=x_case)
        # add columns to df: q-value, scale_factor, and normalize_reward
        mdp_df = self.add_q_scaling(mdp_df)
        #Q-matrix and co.
        states_list, state_action_dict, q_table = self.generate_q_table(mdp_df)

        self.q_table_final, avg_reward_train = self.training_loop(states_list=states_list, state_action_dict=state_action_dict, q_table=q_table, nr_episodes=nr_episodes_train, eval=False)
        if tuning:
            _, avg_reward_eval = self.training_loop(states_list=states_list, state_action_dict=state_action_dict, q_table=self.q_table_final, nr_episodes=nr_episodes_eval, eval=True)

        # self.model = self.q_table_final  # save the final Q-table as the model

        self.model = [self.kmeans_clustering, self.q_table_final]

        # TEST:
        # Normalize silhouette score to [0,1]
        silhouette_norm = (silhouette + 1) / 2

        if self.model_params["pos_rewards"]:
            reward_value = avg_reward_train if not tuning else avg_reward_eval
            silhouette_norm = -silhouette_norm
        else:
            reward_value = -(avg_reward_train if not tuning else avg_reward_eval)

        # reward_norm = sigmoid(reward_value)
        reward_norm = reward_value

        # Weighting factor (hyperparameter)
        alpha = self.model_params.get("silhouette_weight", 0.5)

        # Combined normalized loss
        combined_loss = (1 - alpha) * reward_norm + alpha * silhouette_norm

        print('Silhouette: ', silhouette)
        print('Reward: ', reward_value)

        return combined_loss

        # if self.model_params["pos_rewards"]:
        #     return avg_reward_train if not tuning else avg_reward_eval
        # else:
        #     return - avg_reward_train if not tuning else - avg_reward_eval

    def cluster_data(self, x_case):
        kmeans_clustering = KMeansClustering(model_params=self.model_params)
        silhouette = kmeans_clustering.fit_and_get_loss(x_case=x_case, x_event=None, t=None, prefix_len=None, y=None, weights=None, tuning=False, test_percent=0)
        return kmeans_clustering, silhouette

    def construct_mdp(self, data):
        #NEW
        normalize_reward = self.model_params["normalize_reward"]
        change_zero_reward = self.model_params["change_zero_reward"]
        norm_mdp = self.model_params["norm_mdp"]

        data = data.copy()
        # data["cluster"] = self.models_list_of_dicts[self.stage]["cluster"].labels_
        data["cluster"] = self.kmeans_clustering.forward(x_case=data, x_event=None, t=None, prefix_len=None)
        # state is then activity + cluster
        data["s"] = data["activity"] + " | " + data["cluster"].astype(str)

        # we already have the s, a and reward, now we need the s', which is just the next activity + the next cluster (but if activity != "initiate_application", then the state is just 'end')
        data['next_activity'] = data['activity'].shift(-1)
        data['next_cluster'] = data['cluster'].shift(-1)
        data["next_case_nr"] = data['case_nr'].shift(-1)
        # Apply the condition to generate s'
        data["s'"] = data.apply(
            lambda row: f"{row['next_activity']} | {int(row['next_cluster'])}" if (row['case_nr'] == row["next_case_nr"]) else 'END',
            axis=1
        )
        # Optionally drop the helper columns
        data.drop(['next_activity', 'next_cluster', 'next_case_nr'], axis=1, inplace=True)

        # reward is then the outcome if the activity is not 'initiate_application'
        data["reward"] = data.apply(lambda x: x["outcome"] if x["s'"] == "END" else 0, axis=1)

        # normalize reward column
        if normalize_reward and not norm_mdp:
            minmax_reward = MinMaxScaler()
            nonzero_mask = data["reward"] != 0
            data['scaled_reward'] = 0
            if not change_zero_reward and nonzero_mask.any():
                # Only fit/transform on non-zero rewards
                minmax_reward.fit(data.loc[nonzero_mask, ["reward"]])
                data.loc[nonzero_mask, "scaled_reward"] = minmax_reward.transform(data.loc[nonzero_mask, ["reward"]])
            else:
                data['scaled_reward'] = minmax_reward.fit_transform(data[["reward"]])
            data["reward"] = data['scaled_reward']  # for debug
            data = data.drop(columns=['scaled_reward'])
                
            # # mdp_df["reward"] = np.where(mdp_df["reward"] == 0, 0, minmax_scale.fit_transform(mdp_df[["reward"]]))
            # data['scaled_reward'] = minmax_reward.fit_transform(data[["reward"]])
            # if change_zero_reward:
            #     data['new_reward'] = data['scaled_reward']
            # else:
            #     data['new_reward'] = np.where(mdp_df["reward"] == 0, 0, mdp_df['scaled_reward'])
            # data["reward"] = data['new_reward']  # for debug
            # data = data.drop(columns=['scaled_reward', 'new_reward'])

        # make a new df with the unique combinations of s, a, s'
        mdp_df = data[["s", "a", "s'"]].drop_duplicates()

        # calculate the number of occurrences of each combination of s, a, s', and calculate the average reward for each combination
        counts_s_a_s = data[["s", "a", "s'"]].value_counts().reset_index(name='number_occurrences')
        mdp_df = mdp_df.merge(counts_s_a_s, on=["s", "a", "s'"], how='left')
        # calculate the average reward for each combination of s, a, s'
        avg_rewards = data.groupby(["s", "a", "s'"])["reward"].mean().reset_index(name='reward')
        # merge the average rewards with the mdp_df
        mdp_df = mdp_df.merge(avg_rewards, on=["s", "a", "s'"], how='left')

        # calculate the transition probabilities p_r
        counts_s_a = data[["s", "a"]].value_counts().reset_index(name='count')
        # divide the number of occurrences of each combination of s, a, s' by the number of occurrences of s, a
        mdp_df = mdp_df.merge(counts_s_a, on=["s", "a"], how='left')
        mdp_df["p_r"] = mdp_df["number_occurrences"] / mdp_df["count"]

        print('')

        return mdp_df

    def add_q_scaling(self, mdp_df):
        """
        Add the q-value scaling to the MDP DataFrame.
        """
        scale_factor_type = self.model_params["scale_factor_type"]
        scale_factor_step_smooth = self.model_params["scale_factor_step_smooth"]
        normalize_reward = self.model_params["normalize_reward"]
        change_zero_reward = self.model_params["change_zero_reward"]
        norm_mdp = self.model_params['norm_mdp']
        
        mdp_df['q'] = 0
        # compute total number of occurrences per action
        mdp_df = mdp_df.groupby(['s', "a"]).apply(self.total_occurrences, 'sum_n_occurrences')

        # define scale factor
        # redefine scale factor label for the mdp_df to include the type of scaling
        # "scale_factor" = "scale_factor" + "_" + scale_factor_type
        if scale_factor_type == "none":
            mdp_df["scale_factor"] = 1
        elif scale_factor_type == "linear":
            minmax_scale = MinMaxScaler()
            mdp_df["scale_factor"] = minmax_scale.fit_transform(mdp_df[['sum_n_occurrences']])
        elif scale_factor_type == "smooth":
            w = scale_factor_step_smooth
            # "scale_factor" += "_" + str(w)
            mdp_df["scale_factor"] = mdp_df["sum_n_occurrences"].apply(lambda x: -2 * (math.exp(-x/w)/(1+math.exp(-x/w))) + 1)
        elif scale_factor_type == "step":
            mdp_df["scale_factor"] = mdp_df["sum_n_occurrences"].apply(lambda x: 0 if x <= scale_factor_step_smooth else 1)

        # OLD
        # normalize reward column
        if normalize_reward and norm_mdp:
            minmax_reward = MinMaxScaler()
            nonzero_mask = mdp_df["reward"] != 0
            mdp_df['scaled_reward'] = 0
            if not change_zero_reward and nonzero_mask.any():
                # Only fit/transform on non-zero rewards
                minmax_reward.fit(mdp_df.loc[nonzero_mask, ["reward"]])
                mdp_df.loc[nonzero_mask, "scaled_reward"] = minmax_reward.transform(mdp_df.loc[nonzero_mask, ["reward"]])
            else:
                mdp_df['scaled_reward'] = minmax_reward.fit_transform(mdp_df[["reward"]])
            mdp_df["reward"] = mdp_df['scaled_reward']  # for debug
            mdp_df = mdp_df.drop(columns=['scaled_reward'])
            # minmax_reward = MinMaxScaler()
            # # mdp_df["reward"] = np.where(mdp_df["reward"] == 0, 0, minmax_scale.fit_transform(mdp_df[["reward"]]))
            # mdp_df['scaled_reward'] = minmax_reward.fit_transform(mdp_df[["reward"]])
            # if change_zero_reward:
            #     mdp_df['new_reward'] = mdp_df['scaled_reward']
            # else:
            #     mdp_df['new_reward'] = np.where(mdp_df["reward"] == 0, 0, mdp_df['scaled_reward'])
            # mdp_df["reward"] = mdp_df['new_reward']  # for debug
            # mdp_df = mdp_df.drop(columns=['scaled_reward', 'new_reward'])

        mdp_df = mdp_df.drop(columns=['sum_n_occurrences'])
        return mdp_df
    
    def total_occurrences(self, group, label):
        g = group["number_occurrences"].agg('sum')
        group[label] = g
        return group
    
    def generate_q_table(self, mdp_df):
        """
        q-table is a dict (state) of dict (action)
        each values is a couple: the first value is the q-value, the second value is a dict
        the dict is {next_state: {"p": p, "r": r}}
        in total {s: {a: {"q": 0, "next_state_dict": {s': {"p": p, "r": r}}, "scale_factor": scale_factor}}}
        """
        #extract states
        states_list = np.unique(mdp_df['s'])
        # define action dictionary and q_table
        q_table = {}
        state_action_dict = {}
        for s in states_list:
            state_action_dict[s] = np.unique(mdp_df.loc[mdp_df['s'] == s, ['a']])
            q_table[s] = {}

        # build q_table
        for s, a_list in state_action_dict.items():
            for a in a_list:
                next_state_df = mdp_df.loc[(mdp_df['s'] == s) & (mdp_df['a'] == a),
                                        ["s\'", "p_r", "reward", "scale_factor"]]
                scale_factor = next_state_df["scale_factor"].to_numpy()[0]
                next_state_df = next_state_df.drop(columns=["scale_factor"])
                next_state_dict = self.create_next_state_dict(next_state_df)
                q_table[s][a] = {"q": 0, "next_state_dict": next_state_dict, "scale_factor": scale_factor}

        return states_list, state_action_dict, q_table
    
    def create_next_state_dict(self, next_state_df):
        # create dict {next_state: {"p": p, "r": r, "n_occ": n_occ}}
        next_state_df.set_index("s\'", inplace=True)
        next_state_dict = next_state_df.to_dict('index')
        return next_state_dict

    # def select_action(self, state_action_dict, q_table, state, epsilon):
    #     # select action
    #     if self.random_object.uniform(0, 1) < epsilon:
    #         # exploration: action is taken randomly
    #         action = self.random_object.choice(state_action_dict[state])
    #     else:
    #         # exploitation: action is taken as the argmax
    #         # take the max q-value for that state (q_table[state] is a dict {a: {"q": q, ...}}
    #         max_q_value = max([v["q"] for v in q_table[state].values()])
    #         # select randomly on of the best action
    #         action = self.random_object.choice([k for k, v in q_table[state].items() if v["q"] == max_q_value])
    #     return action

    def select_action(self, state_action_dict, q_table, state, epsilon):
        if state not in q_table:
            # Handle unseen state at inference
            action = 0
            q_values = [0]
            return action, q_values

        if self.random_object.uniform(0, 1) < epsilon:
            actions = state_action_dict[state]  # Needed only for exploration
            # Exploration
            action = self.random_object.choice(actions)
            q_values = np.array([q_table[state][a]["q"] for a in actions], dtype=np.float32)
        else:
            q_values = [v["q"] for v in q_table[state].values()]
            max_q_value = max(q_values)
            action = self.random_object.choice([k for k, v in q_table[state].items() if v["q"] == max_q_value])
        return action, q_values
    
    def compute_policy(self, mdp_df):
        state_set = set(mdp_df["s"])
        max_q_dict = {}
        for s in state_set:
            filtered_df = mdp_df.loc[mdp_df["s"] == s,["s", "q"]]
            max_q_dict[s] = max(filtered_df["q"])
        mdp_df['max_q'] = mdp_df["s"].map(max_q_dict)
        mdp_df["policy"] = np.where(mdp_df["q"] == mdp_df['max_q'], 1, 0)
        mdp_df = mdp_df.drop(columns=['max_q'])
        return mdp_df
    
    def get_states(self, x_case, x_event, t, prefix_len):
        # get the cluster labels for the input data
        state_df = x_case.copy()
        state_df["cluster"] = self.kmeans_clustering.forward(x_case, x_event, t, prefix_len)
        state_df["s"] = state_df["activity"] + " | " + state_df["cluster"].astype(str)

        # drop everything but the state and case_nr
        state_df = state_df[["s", "case_nr"]]

        return state_df
    
    def forward(self, x_case, x_event, t, prefix_len, y, ret_counterfactuals=False):
        """
        Forward pass through the Q-learning model.
        Ensures that action values are aligned to the correct actions and missing ones are -100.
        """

        self.kmeans_clustering = self.model[0]
        self.q_table_final = self.model[1]

        state_df = self.get_states(x_case, x_event, t, prefix_len)

        all_action_values = []

        all_possible_actions = dataset_configs[self.model_params["dataset"]]["actions"]
        all_prev_activities = dataset_configs[self.model_params["dataset"]]["prev_activities"]

        for s in state_df["s"]:
            # Get Q-values for this state
            q_values = self.q_table_final.get(s, {})  # Equivalent to q_table[state]
            # q_values = self.select_action(state_action_dict={}, q_table=self.model, state=s, epsilon=0)[1]
            
            # Determine which "stage" we are in (from the first part of the state)
            prev_activity = s.split(" | ")[0]
            
            # TODO STAGE: check whether this is necessary for the other dataset (bpic17)
            # Find the possible actions for this stage
            # Example: if stage == "initiate_application", then use prev_activities to match it
            possible_actions = []
            for stage in all_prev_activities:
                if prev_activity in stage:
                    possible_actions = all_possible_actions[all_prev_activities.index(stage)]
                    break
            
            if not possible_actions:
                # fallback — if stage not found, assume all actions are possible
                possible_actions = all_possible_actions[0]  # or some default set
            
            # Initialize aligned values with -100
            aligned_values = [-100.0] * len(possible_actions)
            
            # Fill in Q-values where available
            for i, act in enumerate(possible_actions):
                if act in q_values:
                    aligned_values[i] = q_values[act]["q"]
            
            all_action_values.append(aligned_values)

        # Convert to NumPy array (transpose if needed)
        action_values = np.array(all_action_values)

        # action_values = action_values[:, ::-1]

        return action_values

    def training_loop(self, states_list, state_action_dict, q_table, nr_episodes, eval=False):
        loop_penalty = self.model_params["loop_penalty"]
        exceeding_traces_length_penalty = self.model_params["exceeding_traces_length_penalty"]
        max_trace_length = self.model_params["max_trace_length"]
        alpha_max = self.model_params["alpha_max"]
        # alpha_max_to_add = self.model_params["alpha_max_to_add"]
        alpha_min = self.model_params["alpha_min"]
        epsilon = self.model_params["epsilon"] if not eval else 0  # no exploration during evaluation
        gamma = self.model_params["gamma"]

        # alpha_max = min(0.999, alpha_min + alpha_max_to_add)

        reward_returns = 0
        for i in range(1, nr_episodes + 1):
            state = self.random_object.choice(states_list)  # initial state
            action, _ = self.select_action(state_action_dict, q_table, state, epsilon) # initial action
            return_dict = {}
            path = [state]

            reward = 0  # initial reward
            done = False
            alpha = alpha_max - (alpha_max - alpha_min)*(i-1)/(nr_episodes - 1)
            trace_i = 0
            state_action_first_appear_list = []
            while not done:
                # these three variables manage stochastic decision
                summed_probability = 0
                next_state_probability = 0
                choice = self.random_object.uniform(0, 1)
                next_state = ''
                # stochastic decision
                list_of_possibilities = q_table[state][action]["next_state_dict"]
                # list of probabilities transition for the list of possibilities (v is a dict {"p": p, "r": r,...})
                p = np.array([v["p_r"] for x, v in list_of_possibilities.items()])
                p /= p.sum()  # renormalization to avoid machine missing digits
                # stochastically choose next_state
                # next_state = np.random.choice([x for x in list_of_possibilities.keys()], p=p)
                next_state = self.random_object.choices([x for x in list_of_possibilities.keys()], weights=p, k=1)[0]
                reward = list_of_possibilities[next_state]["reward"]

                try:
                    if "END" in next_state:
                        done = True
                        next_action = ""
                    elif next_state not in states_list:
                        # this is the case when the state is not present in the policy model
                        reward = 0
                        done = True
                    else:
                        next_action, _ = self.select_action(state_action_dict, q_table, next_state, epsilon)
                except Exception as e:  # for degug
                    print("Error:")
                    print("i: ", i)
                    print("state: ", state)
                    print("action: ", action)
                    print("choice: ", choice)
                    print("next_state ", next_state)
                    print("next_state_probability ", next_state_probability)
                    print("summed_probability ", summed_probability)

                # Check if is going through a loop
                if next_state in path:
                    reward = loop_penalty
                    done = True

                # Check if trace is too long
                if trace_i > max_trace_length:
                    reward = exceeding_traces_length_penalty
                    done = True

                # add to list of returns
                if (state, action) not in return_dict.keys():
                    return_dict[(state, action)] = 0
                    state_action_first_appear_list.append((state, action))
                return_dict = {k: return_dict[k] + (reward * gamma ** (len(state_action_first_appear_list) - (n+1))) for n, k in enumerate(state_action_first_appear_list)}

                #NEW (old was with return_dict)
                reward_returns += reward

                state = next_state
                action = next_action
                path += [state]
                trace_i += 1

            if not eval:
                # update Q-Table
                for state, action in return_dict.keys():
                    q_value = q_table[state][action]["q"]
                    scale_factor = q_table[state][action]["scale_factor"]
                    # scaled factor takes into accunt the confidence of the transition (number of occurences in the log)
                    # NOTE: we do not use a discount in DTR because we do not incrementally update the Q-values, but rather update them all at once using a fixed dataset, not some experiences
                    q_value_update = return_dict[(state, action)] * scale_factor
                    q_table[state][action]["q"] = (1-alpha) * q_value + alpha * q_value_update  # update q-value

        avg_reward = reward_returns / nr_episodes if nr_episodes > 0 else 0
        return q_table, avg_reward
    
# Deep Learning
def get_dl_model(model_params, n_classes, target_type,):
    if model_params["model_specific"] == "vanilla_nn":
        if target_type == "reg":
            return Vanilla_NN(model_params=model_params)
    else:
        if model_params["model_specific"] == "lstm":
            if target_type == "reg":
                return LSTM(model_params=model_params)
        elif model_params["model_specific"] == "transformer":
            if target_type == "reg":
                return TransformerModel(model_params=model_params)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerModel(nn.Module):
    def __init__(self, model_params, classification=False, n_classes=2):
        super().__init__()

        set_seed(model_params["seed"])

        self.model_params = model_params
        self.dim_x_case = model_params["dim_x_case"]
        # self.dim_x_event = model_params["dim_x_event"]
        to_add = self.model_params["dim_t"] - 1 if self.model_params["dim_t"] > 1 else self.model_params["dim_t"]
        if self.model_params["target"] != "outcome" or self.model_params["action_recomm_method"] == "class" or "T" in self.model_params["method"]:
        # if we are in the propensity score part, we do not include the treatment in the input size
            self.dim_x_event = self.model_params["dim_x_event"]
        else:
            self.dim_x_event = self.model_params["dim_x_event"] + to_add
        self.dim_output = model_params["dim_output"]

        # reuse same naming conventions
        self.nr_lstm_layers = model_params["n_lstm_layers"]  # now # of transformer layers
        self.dim_lstm = model_params["dim_lstm"]  # model dimension for transformer
        # self.nr_dense_layers = model_params["n_dense_layers_in_lstm"]
        self.nr_dense_layers = model_params["n_dense_layers"]
        self.dim_dense = model_params["dim_dense"]
        self.p = model_params["dropout"]

        self.classification = classification
        self.n_classes = n_classes

        # projection from event dim to transformer dim
        self.input_proj = nn.Linear(self.dim_x_event, self.dim_lstm)

        # positional encoding
        self.pos_encoder = PositionalEncoding(self.dim_lstm)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_lstm,
            nhead=model_params.get("nhead", 4),
            dim_feedforward=model_params.get("dim_ff", self.dim_lstm * 4),
            dropout=self.p,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.nr_lstm_layers
        )

        # dense layers
        self.dense_layers = nn.ModuleDict()
        for nr in range(self.nr_dense_layers):
            if nr == 0:
                self.dense_layers[str(nr)] = nn.Linear(self.dim_x_case + self.dim_lstm, self.dim_dense)
            else:
                self.dense_layers[str(nr)] = nn.Linear(self.dim_dense, self.dim_dense)

        # output heads
        self.last_layer = nn.Linear(self.dim_dense, 10)
        if self.classification:
            self.output = nn.Linear(10, self.n_classes - 1 if self.n_classes <= 2 else self.n_classes)
        else:
            self.output = nn.Linear(10, self.dim_output)

        # activations, dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.p)

    def forward(self, x_case, x_process, prefix_len=None, t=None):
        # x_process: (batch, seq_len, dim_x_event)
        x = self.input_proj(x_process)
        x = self.pos_encoder(x)

        # create padding mask if prefix lengths provided
        if prefix_len is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=prefix_len.device)[None, :] >= prefix_len[:, None]
        else:
            mask = None

        x_encoded = self.transformer_encoder(x, src_key_padding_mask=mask)

        # aggregate sequence representation (mean-pooling)
        if prefix_len is not None:
            valid_mask = ~mask
            pooled = (x_encoded * valid_mask.unsqueeze(-1)).sum(1) / prefix_len.unsqueeze(1)
        else:
            pooled = x_encoded.mean(dim=1)

        # concatenate with case-level features
        x_concat = torch.cat((x_case, pooled), dim=1)

        # dense layers
        hidden = nn.Sequential(self.dropout, self.dense_layers["0"], self.relu)(x_concat)
        for nr in range(1, self.nr_dense_layers):
            hidden = nn.Sequential(self.dropout, self.dense_layers[str(nr)], self.relu)(hidden)

        last = nn.Sequential(self.last_layer, self.relu)(hidden)
        output = self.output(last)
        return output

class SoftThreshold(nn.Module):
    def __init__(self, input_dim, sharpness=10.0):
        """
        input_dim: number of features to apply gating to
        sharpness: controls how sharp the threshold is (higher = more step-like)
        """
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(input_dim))  # learnable thresholds
        self.k = nn.Parameter(torch.ones(input_dim) * sharpness)  # learnable sharpness

    def forward(self, x):
        # sigmoid(k * (x - theta)) ≈ 1 if x > theta, 0 if x < theta
        gate = torch.sigmoid(self.k * (x - self.theta))
        return gate

class LSTM(nn.Module):
  def __init__(self, model_params, classification=False, n_classes=2):
    super().__init__()

    set_seed(model_params["seed"])
    
    self.model_params = model_params
    self.dim_x_case = self.model_params["dim_x_case"]
    to_add = self.model_params["dim_t"] - 1 if self.model_params["dim_t"] > 1 else self.model_params["dim_t"]
    if self.model_params["target"] != "outcome" or self.model_params["action_recomm_method"] == "class" or "T" in self.model_params["method"]:
      # if we are in the propensity score part, we do not include the treatment in the input size
      self.dim_x_event = self.model_params["dim_x_event"]
    else:
      self.dim_x_event = self.model_params["dim_x_event"] + to_add
    self.dim_output = self.model_params["dim_output"]
    
    self.nr_lstm_layers = self.model_params["n_lstm_layers"]
    # self.nr_dense_layers = self.model_params["n_dense_layers_in_lstm"]
    self.nr_dense_layers = self.model_params["n_dense_layers"]
    
    self.dim_dense = self.model_params["dim_dense"]
    self.dim_lstm = self.model_params["dim_lstm"]

    self.masked = self.model_params["masked"]
    self.p = self.model_params["dropout"]

    self.classification = classification
    self.n_classes = n_classes

    # self.sharpness = self.model_params["sharpness"]
    
    # INPUT_SIZE_PROCESS CONTAINS THE TREATMENT IF IT IS INCLUDED

    # lstm layers
    if self.nr_lstm_layers > 0:
      self.lstm_layers = nn.ModuleDict()

      for nr in range(self.nr_lstm_layers):
        if nr == 0:
          # self.lstm_layers[str(nr)] = nn.LSTM(self.dim_x_event + 1, self.dim_lstm, 1)
          self.lstm_layers[str(nr)] = nn.LSTM(self.dim_x_event, self.dim_lstm, 1)
        else:
          self.lstm_layers[str(nr)] = nn.LSTM(self.dim_lstm, self.dim_lstm, 1)
    else:
      self.dim_lstm = 0

    # dense layers
    self.dense_layers = nn.ModuleDict()
    for nr in range(self.nr_dense_layers):
      if nr == 0:
        self.dense_layers[str(nr)] = nn.Linear(self.dim_x_case + self.dim_lstm, self.dim_dense)
        #self.batchnorm1 = nn.BatchNorm1d(width, affine=False)
      else:
        self.dense_layers[str(nr)] = nn.Linear(self.dim_dense, self.dim_dense)

    # # outputs
    # self.last_layer = nn.Linear(self.dim_dense, 10)
    # self.output_mean = nn.Linear(10, self.dim_output)

    # output layers
    self.last_layer = nn.Linear(self.dim_dense, 10)
    if self.classification:
        self.output = nn.Linear(10, self.n_classes - 1 if self.n_classes <= 2 else self.n_classes)
    else:
        self.output = nn.Linear(10, self.dim_output)

    # parameter-free layers
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(self.p)


  def forward(self, x_case, x_process, prefix_len=None, t=None):
    # X_PROCESS CONTAINS THE TREATMENT IF IT IS INCLUDED
    x = x_process

    # lstm layers
    if self.nr_lstm_layers > 0:
      x = x.transpose(0, 2)
      x = x.transpose(1, 2)
      if self.masked:
        # Sort and remember the original sorting
        x = nn.utils.rnn.pack_padded_sequence(x, lengths=prefix_len, enforce_sorted=False)

      outputs, (h, c) = self.lstm_layers[str(0)](x)
      ### !!! ### this dropout is only for standard LSTMs
      # outputs = self.dropout(outputs)

      for nr in range(1, self.nr_lstm_layers):
        outputs, (h, c) = self.lstm_layers[str(nr)](outputs, (h, c))
        # outputs = self.dropout(outputs)

      outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
      # grab the corresponding correct outputs (with prefix_len)
      outputs = outputs[prefix_len.long() - 1, np.arange(len(prefix_len))]

      # concatenate lstm output with case variables
      x_concat = torch.cat((x_case, outputs), 1)
    else:
      x_concat = x_case

      x_concat = self.relu(x_concat)

    # calculate dense layers
    if self.nr_dense_layers > 0:
      hidden = nn.Sequential(self.dropout, self.dense_layers[str(0)], self.relu)(x_concat)
      for nr in range(1, self.nr_dense_layers):
        hidden = nn.Sequential(self.dropout, self.dense_layers[str(nr)], self.relu)(hidden)
        # x = nn.Sequential(self.dropout, self.layer1, self.relu)(x) #, self.batchnorm1)(x

    # Output
    last = nn.Sequential(self.last_layer, self.relu)(hidden)
    output = self.output(last)

    return output  # raw logits for classification; raw values for regression

class Vanilla_NN(nn.Module):
    def __init__(self, model_params, classification=False, n_classes=2):
        """
        Simple fully-connected network (MLP), consistent with LSTM parameters.
        """
        super().__init__()

        set_seed(model_params["seed"])

        self.model_params = model_params
        self.classification = classification
        self.n_classes = n_classes

        # Input dimension
        to_add = self.model_params["dim_t"] - 1 if self.model_params["dim_t"] > 1 else self.model_params["dim_t"]
        if (self.model_params["target"] != "outcome" or self.model_params["action_recomm_method"] == "class" or "T" in self.model_params["method"]):
            self.dim_x_case = self.model_params["dim_x_case"]
        else:
            self.dim_x_case = self.model_params["dim_x_case"] + to_add

        # Dense layers
        self.nr_dense_layers = self.model_params["n_dense_layers"]
        self.dim_dense = self.model_params["dim_dense"]
        self.dropout_p = self.model_params.get("dropout", 0.0)

        # Determine output dimension
        if not self.classification:
            self.dim_output = self.model_params.get("dim_output", 1)
        elif self.n_classes == 2:
            self.dim_output = 1
        else:
            self.dim_output = self.n_classes

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_p)

        # self.sharpness = self.model_params["sharpness"]

        self.model = self._build_model()

    def _build_model(self):
        layers = []
        # First layer
        layers.append(nn.Linear(self.dim_x_case, self.dim_dense))
        layers.append(self.relu)
        layers.append(self.dropout)

        # Hidden layers
        for _ in range(self.nr_dense_layers - 1):
            layers.append(nn.Linear(self.dim_dense, self.dim_dense))
            layers.append(self.relu)
            layers.append(self.dropout)

        # Output layer
        layers.append(nn.Linear(self.dim_dense, self.dim_output))
        # No activation on output (loss function handles it)
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DLCausalRegressor:
    def __init__(self, model_params):
        """
        Neural Network-based causal regressor compatible with external train_dl().
        Supports both S- and T-learners, and handles LSTM or MLP architectures.
        """
        self.model_params = model_params
        self.single_model = (
            "S" in model_params["method"] and model_params["target"] == "outcome"
        )
        self.target = model_params["target"]
        self.n_classes = model_params["dim_t"] if model_params["dim_t"] > 1 else 2

        base_model = get_dl_model(
            model_params=model_params, n_classes=self.n_classes, target_type="reg"
        )

        if self.single_model:
            self.model = deepcopy(base_model)
        else:
            self.model = [deepcopy(base_model) for _ in range(self.n_classes)]

        self.criterion = torch.nn.MSELoss(reduction="mean")

    def get_loss(self, x_case, x_event, t, prefix_len, y, weights=None, set_eval=True):
        """
        Compute batch loss for training.
        """
        if set_eval:
            # set model to evaluation mode
            if self.single_model:
                self.model.eval()
            else:
                for m in self.model:
                    m.eval()
        else:
            if self.single_model:
                self.model.train()
            else:
                for m in self.model:
                    m.train()

        y = y.squeeze(1) if len(y.shape) == 2 else y

        if self.single_model:
            t = t[:, 1:] if self.n_classes > 2 else t
            if self.model_params["model_specific"] == "lstm" or self.model_params["model_specific"] == "transformer":

                # Inject treatment into the event sequence
                t_adjusted = torch.zeros(
                    size=(t.shape[0], self.model_params["dim_t"] - 1, x_event.shape[2]),
                )
                t_adjusted[range(t.shape[0]), :, prefix_len.long() - 1] = t
                x_event_with_t = torch.cat((x_event, t_adjusted), dim=1) if self.target == "outcome" else x_event

                y_ = self.model.forward(
                    x_case=x_case, x_process=x_event_with_t, prefix_len=prefix_len
                )
                y_ = y_.squeeze(1) if len(y_.shape) == 2 else y_
                loss = torch.nn.functional.mse_loss(y_, y) if not set_eval else torch.mean(torch.abs(y - y_))
            else:
                # Vanilla NN case
                x_with_t = torch.cat((x_case, t), dim=1) if self.target == "outcome" else x_case
                y_ = self.model.forward(x_with_t)
                y_ = y_.squeeze(1) if len(y_.shape) == 2 else y_
                loss = torch.nn.functional.mse_loss(y_, y) if not set_eval else torch.mean(torch.abs(y - y_))
        else:
            # T-learner case: separate model per treatment arm
            loss = [] # NOTE: loss is a list when a T-learner is used
            for arm_idx, model in enumerate(self.model):
                # t is shaped like this [ [0, 0, 1] ] if there are more than 2 classes, else [ [0], [1], ... ]
                if t.ndim == 2 and t.shape[1] > 1:
                    # Multi-class, one-hot encoded
                    mask = (t[:, arm_idx] == 1)
                else:
                    # Binary or integer-encoded
                    mask = (t.flatten() == arm_idx)

                if mask.sum() == 0:
                    loss.append(None)
                    continue

                if self.model_params["model_specific"] == "lstm" or self.model_params["model_specific"] == "transformer":
                    if self.target == "outcome":
                        x_case_adjusted = x_case[mask]
                        x_event_adjusted = x_event[mask]
                        prefix_len_adjusted = prefix_len[mask]
                        y_adjusted = y[mask]
                    else:
                        x_case_adjusted = x_case
                        x_event_adjusted = x_event
                        prefix_len_adjusted = prefix_len
                        y_adjusted = y[:, arm_idx]

                    y_ = model.forward(
                        x_case=x_case_adjusted,
                        x_process=x_event_adjusted,
                        prefix_len=prefix_len_adjusted,
                    )
                    y_ = y_.squeeze(1) if len(y_.shape) == 2 else y_
                    to_append = torch.nn.functional.mse_loss(y_, y) if not set_eval else torch.mean(torch.abs(y - y_))
                    loss.append(to_append)
                else:
                    if self.target == "outcome":
                        x_case_adjusted = x_case[mask]
                        y_adjusted = y[mask]
                    else:
                        x_case_adjusted = x_case
                        y_adjusted = y[:, arm_idx]

                    y_ = model.forward(x_case_adjusted)
                    y_ = y_.squeeze(1) if len(y_.shape) == 2 else y_
                    to_append = torch.nn.functional.mse_loss(y_, y) if not set_eval else torch.mean(torch.abs(y - y_))
                    loss.append(to_append)

        return loss

    def forward(self, x_case, x_event, t, prefix_len, y=None, ret_counterfactuals=False, set_eval=True):
        """
        Forward pass for inference or counterfactual prediction.
        """
        if set_eval:
            # set model to evaluation mode
            if self.single_model:
                self.model.eval()
            else:
                for m in self.model:
                    m.eval()
        else:
            if self.single_model:
                self.model.train()
            else:
                for m in self.model:
                    m.train()

        preds = []
        if self.single_model and self.target == "outcome":
            for treatment in range(self.n_classes):
                if self.model_params["model_specific"] == "lstm" or self.model_params["model_specific"] == "transformer":
                    # treatment = treatment[:, 1:]

                    if treatment == 0:
                        treatment_vec = torch.zeros(self.model_params["dim_t"] - 1)
                    else:
                        treatment_vec = torch.zeros(self.model_params["dim_t"] - 1)
                        treatment_vec[treatment - 1] = 1

                    t_artificial = torch.zeros(
                        size=(x_event.shape[0], self.model_params["dim_t"] - 1, x_event.shape[2]),
                    )
                    t_artificial[range(x_event.shape[0]), :, prefix_len.long() - 1] = treatment_vec
                    x_event_with_t = torch.cat((x_event, t_artificial), dim=1)
                    preds_t = self.model.forward(
                        x_case=x_case, x_process=x_event_with_t, prefix_len=prefix_len
                    )
                    preds_t = preds_t.squeeze(1) if len(preds_t.shape) == 2 else preds_t
                else:
                    T_artificial = (
                        torch.zeros((len(x_case), 1))
                        if self.n_classes == 2
                        else torch.zeros((len(x_case), self.n_classes - 1))
                    )
                    if self.n_classes > 2 and treatment != 0:
                        T_artificial[:, treatment - 1] = 1
                    else:
                        T_artificial[:] = treatment
                    x_with_t = torch.cat((x_case, T_artificial), dim=1)
                    preds_t = self.model.forward(x_with_t)
                    preds_t = preds_t.squeeze(1) if len(preds_t.shape) == 2 else preds_t
                preds.append(preds_t)
        else:
            for arm_idx, model in enumerate(self.model):
                if self.model_params["model_specific"] == "lstm" or self.model_params["model_specific"] == "transformer":
                    preds_t = model.forward(x_case=x_case, x_process=x_event, prefix_len=prefix_len)
                else:
                    preds_t = model.forward(x_case)
                preds_t = preds_t.squeeze(1) if len(preds_t.shape) == 2 else preds_t
                preds.append(preds_t)

        preds = torch.stack(preds).squeeze()
        return preds if ret_counterfactuals else preds.mean(0)

class LSTM_Y_SLearner():
    def __init__(self, model_params):
            self.model_params = model_params
            self.model = LSTM(model_params=self.model_params)
            
    def forward(self, x_case, x_event, t, prefix_len, y, ret_counterfactuals=False, eval=False):
        if eval:
            # set model to evaluation mode
            self.model.eval()
        y_ = []
        if ret_counterfactuals:
            # TODO: check if this is correct (when returning counterfactuals)
            for i in range(max(self.model_params["dim_t"], 2)):
                t_adjusted = torch.zeros(size=(t.shape[0], self.model_params["dim_t"], x_event.shape[2]))
                if self.model_params["dim_t"] > 1:
                    t_adjusted[range(t.shape[0]), i, prefix_len.long() - 1] = 1
                else:
                    t_adjusted[range(t.shape[0]), :, prefix_len.long() - 1] = i
                x_event_with_t = torch.cat((x_event, t_adjusted), dim=1)
                instance_y_ = self.model.forward(x_case=x_case, x_process=x_event_with_t, prefix_len=prefix_len)
                instance_y_ = instance_y_.squeeze(1) if len(instance_y_.shape) == 2 else instance_y_
                y_.append(instance_y_)
        else:
            t_adjusted = torch.zeros(size=(t.shape[0], self.model_params["dim_t"], x_event.shape[2]))
            t_adjusted[range(t.shape[0]), t.long(), prefix_len.long() - 1] = t
            x_event_with_t = torch.cat((x_event, t_adjusted), dim=1)
            y_ = self.model.forward(x_case=x_case, x_process=x_event_with_t, prefix_len=prefix_len)
            y_ = y_.squeeze(1) if len(y_.shape) == 2 else y_
        return y_

    def get_loss(self, x_case, x_event, t, prefix_len, y, weights):
        t_adjusted = torch.zeros(size=(t.shape[0], self.model_params["dim_t"], x_event.shape[2]))
        t_adjusted[range(t.shape[0]), :, prefix_len.long() - 1] = t
        x_event_with_t = torch.cat((x_event, t_adjusted), dim=1)
        y_ = self.model.forward(x_case=x_case, x_process=x_event_with_t, prefix_len=prefix_len)
        y_ = y_.squeeze(1) if len(y_.shape) == 2 else y_
        loss_y = torch.nn.functional.mse_loss(y_, y)
        return loss_y
  
class Vanilla_NN_Y_SLearner():
    def __init__(self, model_params):
            self.model_params = model_params
            self.model = Vanilla_NN(model_params=self.model_params)
        
    def forward(self, x_case, x_event, t, prefix_len, y, ret_counterfactuals=False):
        y_ = []
        if ret_counterfactuals:
            for i in range(max(self.model_params["dim_t"], 2)):
                t_adjusted = torch.zeros(size=(t.shape[0], self.model_params["dim_t"]))
                if self.model_params["dim_t"] > 1:
                    t_adjusted[range(t.shape[0]), i] = 1
                else:
                    t_adjusted[range(t.shape[0]), :] = i
                x_with_t = torch.cat((x_case, t_adjusted), dim=1)
                instance_y_ = self.model.forward(x_with_t)
                instance_y_ = instance_y_.squeeze(1) if len(instance_y_.shape) == 2 else instance_y_
                y_.append(instance_y_)
        else:
            x_with_t = torch.cat((x_case, t), dim=1)
            y_ = self.model.forward(x_with_t)
            y_ = y_.squeeze(1) if len(y_.shape) == 2 else y_
        return y_

    def get_loss(self, x_case, x_event, t, prefix_len, y, weights):
        x_with_t = torch.cat((x_case, t), dim=1)
        y_ = self.model.forward(x_with_t)
        y_ = y_.squeeze(1) if len(y_.shape) == 2 else y_
        loss_y = torch.nn.functional.mse_loss(y_, y)
        # zero_tensor = torch.zeros_like(y)
        # loss_y = torch.nn.functional.mse_loss(zero_tensor, y)
        return loss_y
    
class LSTM_Opt():
    def __init__(self, model_params):
        self.model_params = model_params
        self.n_classes = model_params["dim_t"] if model_params["dim_t"] > 1 else 2  # If binary treatment, we use one-hot encoding, so dim_t = 1
        self.model = LSTM(model_params=self.model_params, classification=True, n_classes=self.n_classes)
    
    def forward(self, x_case, x_event, t, prefix_len, y, ret_counterfactuals=False, dataset_ps=None):
        # do not require grad
        with torch.no_grad():
            y_ = self.model.forward(x_case=x_case, x_process=x_event, prefix_len=prefix_len)
            y_ = y_.squeeze(1) if len(y_.shape) == 2 and self.n_classes == 2 else y_

            if self.n_classes > 2:
                # Multi-class: apply softmax to get class probabilities
                proba = torch.softmax(y_, dim=1)
            else:
                # Binary: apply sigmoid to get probability of class 1
                proba = torch.sigmoid(y_).unsqueeze(1)  # [batch_size, 1]
                proba = torch.cat([1 - proba, proba], dim=1)  # Convert to [P(class=0), P(class=1)]

            return proba
            # return proba.T  # Match XGBoost style: shape [n_classes, batch_size]
    
    def get_loss(self, x_case, x_event, t, prefix_len, y, weights, infer=False):
        # y: true labels (either one-hot or indices)
        y = torch.argmax(y, dim=1) if self.n_classes > 2 else y  # convert one-hot to index if needed

        # Forward pass
        y_ = self.model.forward(x_case=x_case, x_process=x_event, prefix_len=prefix_len)
        y_ = y_.squeeze(1) if len(y_.shape) == 2 and self.n_classes == 2 else y_

        if self.n_classes > 2:
            # Cross-entropy loss (no reduction)
            loss_per_sample = torch.nn.functional.cross_entropy(y_, y, reduction='none')
            pred_classes = torch.argmax(y_, dim=1)  # predicted class index per sample
        else:
            # Binary case
            loss_per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
                y_, y.float(), reduction='none'
            )
            pred_classes = (torch.sigmoid(y_) >= 0.5).long().view(-1)

        # Select per-sample weight from (batch_size, n_classes)
        selected_weights = weights[torch.arange(weights.size(0)), pred_classes]

        # Compute weighted mean loss
        weighted_loss = torch.mean(loss_per_sample * selected_weights.to(loss_per_sample.device))

        return weighted_loss
    
    # def get_loss(self, x_case, x_event, t, prefix_len, y, weights):
    #     weights = weights.to(y.device).view(-1).detach()  # <-- Critical fix
    #     # take the argmax if n_classes > 2, otherwise use t directly
    #     y = torch.argmax(y, dim=1) if self.n_classes > 2 else y
    #     y_ = self.model.forward(x_case=x_case, x_process=x_event, prefix_len=prefix_len)
    #     y_ = y_.squeeze(1) if len(y_.shape) == 2 else y_
    #     if self.n_classes > 2:
    #         loss_per_sample = torch.nn.functional.cross_entropy(y_, y, reduction='none')
    #     else:
    #         # For binary treatment, we use binary cross-entropy
    #         loss_per_sample = torch.nn.functional.binary_cross_entropy_with_logits(y_, y.float(), reduction='none')
    #     # apply weights
    #     weighted_loss = torch.mean(loss_per_sample * weights)
    #     return weighted_loss

class Vanilla_NN_Opt():
    def __init__(self, model_params):
        self.model_params = model_params
        self.n_classes = model_params["dim_t"] if model_params["dim_t"] > 1 else 2  # If binary treatment, we use one-hot encoding, so dim_t = 1
        self.model = Vanilla_NN(model_params=self.model_params, classification=True, n_classes=self.n_classes)
    
    def forward(self, x_case, x_event, t, prefix_len, y, ret_counterfactuals=False, dataset_ps=None):
        # do not require grad
        with torch.no_grad():
            y_ = self.model.forward(x_case)
            y_ = y_.squeeze(1) if len(y_.shape) == 2 and self.n_classes == 2 else y_

            if self.n_classes > 2:
                # Multi-class: apply softmax to get class probabilities
                proba = torch.softmax(y_, dim=1)
            else:
                # Binary: apply sigmoid to get probability of class 1
                proba = torch.sigmoid(y_).unsqueeze(1)  # [batch_size, 1]
                proba = torch.cat([1 - proba, proba], dim=1)  # Convert to [P(class=0), P(class=1)]

            return proba
            # return proba.T  # Match XGBoost style: shape [n_classes, batch_size]

    def get_loss(self, x_case, x_event, t, prefix_len, y, weights, infer=False):
        # y: ground truth labels (either one-hot or index form)
        # weights: shape (batch_size, n_classes), contains weight per class per sample
        
        # Convert y to class indices if needed
        y = torch.argmax(y, dim=1) if self.n_classes > 2 else (
            y.squeeze(1) if len(y.shape) == 2 else y
        )

        # Forward pass
        y_ = self.model.forward(x_case)
        y_ = y_.squeeze(1) if len(y_.shape) == 2 and self.n_classes == 2 else y_

        if self.n_classes > 2:
            # Multi-class: CE loss without reduction
            loss_per_sample = torch.nn.functional.cross_entropy(y_, y, reduction='none')
            pred_classes = torch.argmax(y_, dim=1)  # predicted class per sample
        else:
            # Binary: BCE loss without reduction
            loss_per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
                y_, y.float(), reduction='none'
            )
            pred_classes = (torch.sigmoid(y_) >= 0.5).long().view(-1)

        # Select the weight for each sample based on predicted class
        selected_weights = weights[torch.arange(weights.size(0)), pred_classes]

        # Apply weights and compute mean loss
        weighted_loss = torch.mean(loss_per_sample * selected_weights.to(loss_per_sample.device))

        return weighted_loss