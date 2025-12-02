import os
from copy import deepcopy
from functools import reduce
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from hyperopt import STATUS_OK, Trials, fmin, tpe, space_eval
from functools import partial

from config.config import space_dict, make_kmeans_q_space
from src.methods.scope.scope_functions import SCOPEFunctions
from src.methods.separate.separate_functions import SeparateFunctions
from src.methods.kmeans_q.kmeans_q_functions import KMeansQFunctions
from src.utils.model_tools.model_training import ModelTrainer
from src.utils.model_tools.model_eval import ModelEval
from src.utils.mini_tools import (
    create_splits,
    get_model_params_list_of_dicts,
    save_data, load_data
)

class Method():
    def __init__(self, args, method, prepped_data_dict, best_model_params_list_of_dicts=None, iter=0):
        to_add = ''
        if args.dataset == 'bpic17':
            to_add = os.path.join("bpic17", str(args.n_stages))
        self.RESULTS_FOLDER = os.path.join("res", to_add, str(args.train_size), str(int(100 * args.delta)))
        self.PATH_BEGIN = str(args.train_size) + "_" + str(int(100*args.delta)) + "_"

        self.iter = iter
        self.method = method
        self.args = args
        self.prepped_data_dict = prepped_data_dict

        if best_model_params_list_of_dicts is None:
            self.model_params_list_of_dicts = get_model_params_list_of_dicts(method=self.method, args=args, prep_utils=prepped_data_dict["utils"])
        else:
            self.model_params_list_of_dicts = deepcopy(best_model_params_list_of_dicts)

        self.models_list_of_dicts = [{} for _ in range(self.args.n_stages)]

        if "dtr" in self.method:
            self.method_functions = SCOPEFunctions(model_params_list_of_dicts=self.model_params_list_of_dicts)
        elif "separate" in self.method:
            self.method_functions = SeparateFunctions(model_params_list_of_dicts=self.model_params_list_of_dicts)
        elif self.method == "kmeans_q":
            self.method_functions = KMeansQFunctions(model_params_list_of_dicts=self.model_params_list_of_dicts)
        
    def run(self, tuning=False):
        # NOTE: here we start to go over the stages in reverse order for the backward induction
        for stage in range(len(self.model_params_list_of_dicts) - 1, -1, -1):
            self.stage = stage
            model_params_dict = self.model_params_list_of_dicts[stage]
            for target, model_params in model_params_dict.items():
                print(f"    Stage: {stage}, Target: {target}")

                # Init variables
                if model_params == "nope": continue
                to_add_path = "tuning" if tuning else (str(self.iter) + "_training")
                to_add_folder = "tuning" if tuning else "training"
                to_add_cross_fitting = "cross_fitted_" if "dtr" in self.method and self.args.cross_fitting else ""
                model_params["savepath_ps_model"] = os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), to_add_folder, self.PATH_BEGIN + self.method + "_ps_" + model_params["ps_model_params"]["model_specific"] + "_" + str(stage) + "_" + to_add_cross_fitting + to_add_path  + "_model")
                model_params["seed"] = 42 + 5*self.iter
                self.model_params = model_params

                if (target in self.args.already_tuned_list and tuning) or (target in self.args.already_trained_list and not tuning) or (self.args.already_tuned and tuning) or (self.args.already_trained and not tuning):
                    # load the best model and params
                    self.best_model = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), to_add_folder, self.PATH_BEGIN + self.method + "_" + target + "_" + model_params["model_specific"] + "_" + str(stage) + "_" + to_add_cross_fitting + to_add_path + "_model"), is_state_dict=(self.model_params["model_category"] == "dl" and "S" in self.method and target == "outcome"))
                    best_params = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), to_add_folder, self.PATH_BEGIN + self.method + "_" + target + "_" + model_params["model_specific"] + "_" + str(stage) + "_" + to_add_cross_fitting + to_add_path + "_params"))
                    model_params.update(best_params)
                else:
                    data_train_list_ps, data_infer_list_ps = None, None
                    data_train_list_prev_ps, data_infer_list_prev_ps = None, None
                    data_train_list_prev_outcome, data_infer_list_prev_outcome = None, None
                    if "dtr" in self.method:
                        if (target == "effect"):
                            # still pass the class model_params to get the correct splits
                            data_train_list_ps, data_infer_list_ps = create_splits(data_train_list=self.prepped_data_dict["train"][model_params["ps_model_params"]["encoding"]], data_infer_list=self.prepped_data_dict["infer"][model_params["ps_model_params"]["encoding"]], model_params=model_params)
                        if (target == "outcome" or target == "effect") and stage < len(self.model_params_list_of_dicts) - 1:
                            if self.model_params_list_of_dicts[stage + 1]["ps"] != "nope":
                                data_train_list_prev_ps, data_infer_list_prev_ps = create_splits(data_train_list=self.prepped_data_dict["train"][model_params["prev_ps_model_params"]["encoding"]], data_infer_list=self.prepped_data_dict["infer"][model_params["prev_ps_model_params"]["encoding"]], model_params=model_params)
                            data_train_list_prev_outcome, data_infer_list_prev_outcome = create_splits(data_train_list=self.prepped_data_dict["train"][model_params["prev_outcome_model_params"]["encoding"]], data_infer_list=self.prepped_data_dict["infer"][model_params["prev_outcome_model_params"]["encoding"]], model_params=model_params)
                            # BACKWARD INDUCTION:
                            data_train_list_prev_outcome[stage + 1]["Y"] = prev_outcomes_train
                            if data_infer_list_prev_outcome is not None and prev_outcomes_infer is not None:
                                data_infer_list_prev_outcome[stage + 1]["Y"] = prev_outcomes_infer

                    # data_for_other_models
                    data_lists_for_other_models = {"ps": {"train": data_train_list_ps, "infer": data_infer_list_ps},
                                                "prev_ps": {"train": data_train_list_prev_ps, "infer": data_infer_list_prev_ps},
                                                "prev_outcome": {"train": data_train_list_prev_outcome, "infer": data_infer_list_prev_outcome}}

                    # Split data correctly
                    data_train_list, data_infer_list = create_splits(data_train_list=self.prepped_data_dict["train"][model_params["encoding"]], data_infer_list=self.prepped_data_dict["infer"][model_params["encoding"]], model_params=model_params)

                    # Prepare data if needed (e.g., to calculate the targets of outcome in stage 0)
                    self.data_train, self.data_infer, self.weights_train, self.weights_infer, self.data_train_ps, self.data_infer_ps = self.method_functions.prepare(data_train_list=data_train_list, data_infer_list=data_infer_list, stage=self.stage, model_params=model_params, data_lists_for_other_models=data_lists_for_other_models)

                    if "dtr" in self.method and (target == "outcome") and stage > 0:
                        # BACKWARD INDUCTION:
                        prev_outcomes_train = deepcopy(self.data_train["Y"])
                        if self.data_infer is not None:
                            prev_outcomes_infer = deepcopy(self.data_infer["Y"])

                    # Train or Tune
                    if tuning:
                        self.best_loss_infer = float('inf')
                        self.trials = Trials()

                        # SET NUM OF TRIALS FOR TUNING
                        if not self.args.big_tuning:
                            num_tuning_evals = 3
                        elif model_params["model_category"] == "rl":
                            # KMeans and Q-learner get tuned simultaneously, so do twice the number of evals as usual
                            num_tuning_evals = 2*self.args.max_num_tuning_evals if (model_params["model_category"] != "dl" and self.args.big_data) else 2*42
                        else:
                            num_tuning_evals = self.args.max_num_tuning_evals if (model_params["model_category"] != "dl" and self.args.big_data) else 42
                        
                        algo = partial(tpe.suggest, n_startup_jobs=5)

                        model_space = deepcopy(space_dict[model_params["model_specific"]]) if model_params["method"] != "kmeans_q" else make_kmeans_q_space(feature_names=self.data_train.columns.tolist(), args=self.args)
                        best_params = fmin(fn=self._objective,
                                        space=model_space,
                                        algo=algo,
                                        max_evals=num_tuning_evals,
                                        trials=self.trials,
                                        rstate=np.random.default_rng(model_params["seed"]),
                                        show_progressbar=False)
                        # Update according to the model_space
                        best_params = space_eval(model_space, best_params)
                        model_params.update(best_params)
                    else:
                        model_trainer = ModelTrainer(args=self.args, data_train=self.data_train, data_infer=self.data_infer, weights_train=self.weights_train, weights_infer=self.weights_infer, model_params=model_params, data_train_ps=self.data_train_ps, data_infer_ps=self.data_infer_ps)
                        model_trainer.train()
                        self.best_model = model_trainer.get_model()

                # Get the best model
                self.models_list_of_dicts[self.stage][target] = self.best_model

                # IMPORTANT NOTE: Update
                self.method_functions.models_list_of_dicts = deepcopy(self.models_list_of_dicts)
                self.method_functions.model_params_list_of_dicts = deepcopy(self.model_params_list_of_dicts)

                # Save
                save_data(self.models_list_of_dicts[self.stage][target], os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), to_add_folder, self.PATH_BEGIN + self.method + "_" + target + "_" + self.model_params["model_specific"] + "_" + str(stage) + "_" + to_add_cross_fitting + to_add_path  + "_model"))
                save_data(self.model_params_list_of_dicts[self.stage][target], os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), to_add_folder, self.PATH_BEGIN + self.method + "_" + target + "_" + self.model_params["model_specific"] + "_" + str(stage) + "_" + to_add_cross_fitting + to_add_path  + "_params"))
                print('\n')

    def _objective(self, params):
        # Ensure alpha_max > alpha_min
        if self.model_params["method"] == "kmeans_q":
            params["alpha_max"] = params["alpha_min"] + 0.0001 if params["alpha_max"] < params["alpha_min"] + 0.0001 else params["alpha_max"]
        model_params = deepcopy(self.model_params)
        model_params.update(params)
        self.method_functions.model_params.update(model_params)

        model_trainer = ModelTrainer(args=self.args, data_train=self.data_train, data_infer=self.data_infer, weights_train=self.weights_train, weights_infer=self.weights_infer, model_params=model_params, data_train_ps=self.data_train_ps, data_infer_ps=self.data_infer_ps)
        model_trainer.train(tuning=True)
        loss_infer = model_trainer.best_loss_infer
        if loss_infer < self.best_loss_infer:
            self.best_loss_infer = loss_infer
            if model_params["model_category"] == "ml":
                # Train the model once more on full set with the best parameters because we used cross-validation
                model_trainer.train(tuning=False)
            self.best_model = model_trainer.get_model()
            print('Best params:', model_params)

        return {'loss': loss_infer, 'status': STATUS_OK}
    
    def eval(self, preps_maps, dfs_map):
        bank_profit, random_profit, random_uplift, optimal_profit, optimal_uplift = self.get_bank_optimal_random_results(dfs_map=dfs_map)
        
        # Getting the final target and model_params, so the model that actually recommends actions
        for target, model_params in reversed(list(self.model_params_list_of_dicts[0].items())):
            if model_params == "nope": continue
            final_target, final_model_category, final_model_specific = target, model_params["model_category"], model_params["model_specific"]
            to_add_cross_fitting = "cross_fitted_" if "dtr" in self.method and self.args.cross_fitting else ""
            break # model has been found

        to_add_model_specific = final_model_specific + "_" if (final_model_specific != "kmeans_q" and final_model_specific != "xgb" and final_model_specific != "lstm" and final_model_specific != "vanilla_nn") else ""

        if self.method in self.args.already_evaluated_list or self.args.already_evaluated:
            # jus load the results
            profit = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), "eval", self.PATH_BEGIN + self.method + "_" + final_target + "_" + final_model_category + "_" + to_add_model_specific + str(self.iter) + "_" + to_add_cross_fitting + "profit_eval"))
            final_df = None
            # final_df = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), "eval", self.PATH_BEGIN + self.method + "_" + final_target + "_" + final_model_category + "_" + to_add_model_specific + str(self.iter) + "_" + to_add_cross_fitting + "df_eval"))
            uplift = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), "eval", self.PATH_BEGIN + self.method + "_" + final_target + "_" + final_model_category + "_" + to_add_model_specific + str(self.iter) + "_" + to_add_cross_fitting + "uplift_eval"))
        else:
            decisions = {}  # Store decisions for each stage
            case_nrs = list(range(self.args.test_size))
            # IMPORTANT NOTE: for q-learning, we train using 1 stage (0), but for evaluation we need to go through all stages, since the action in stage 0 decides the action in stage 1
            # So whenever we select something of a model, we select it by index_to_select, which is not the same as the stage when the method is "kmeans_q" (which only has one stage for training, so we always select 0)
            action_df = None
            for stage in range(self.args.n_stages):
                index_to_select = stage if self.method != "kmeans_q" else 0  # KMeansQ only has one stage for training, so we always have 0 to select the model
                model_params = self.model_params_list_of_dicts[index_to_select][final_target].copy() 
                collated_prep = self.get_data_given_prev_actions(stage=stage, preps_maps=preps_maps, model_params=model_params, target=final_target, index_to_select=index_to_select, prev_decisions=decisions)
                action_df = self.get_actions_recommended_current_stage(stage=stage, target=final_target, index_to_select=index_to_select, model_params=model_params, collated_prep=collated_prep, case_nrs=case_nrs, to_add_cross_fitting=to_add_cross_fitting)
                decisions[stage] = action_df

            profit, final_df = self.calculate_proft_of_decisions(decisions=decisions, dfs_map=dfs_map)
            # calculate the extra percentage profit compared to the bank
            uplift = (profit - bank_profit) / abs(bank_profit) * 100 if bank_profit != 0 else 0
            print(f"                Random uplift: {random_uplift:.2f}%; Optimal uplift {optimal_uplift:.2f}%; Uplift: {uplift:.2f}%")


            # Save the results
            save_data(decisions, os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), "eval", self.PATH_BEGIN + self.method + "_" + final_target + "_" + final_model_category + "_" + to_add_model_specific + str(self.iter) + "_" + to_add_cross_fitting + "decisions_eval"))
            save_data(profit, os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), "eval", self.PATH_BEGIN + self.method + "_" + final_target + "_" + final_model_category + "_" + to_add_model_specific + str(self.iter) + "_" + to_add_cross_fitting + "profit_eval"))
            # save_data(final_df, os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), "eval", self.PATH_BEGIN + self.method + "_" + final_target + "_" + final_model_category + "_" + to_add_model_specific + str(self.iter) + "_" + to_add_cross_fitting + "df_eval"))
            save_data(uplift, os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), "eval", self.PATH_BEGIN + self.method + "_" + final_target + "_" + final_model_category + "_" + to_add_model_specific + str(self.iter) + "_" + to_add_cross_fitting + "uplift_eval"))

        return uplift, profit, final_df

    def get_bank_optimal_random_results(self, dfs_map):
        # BANK POLICY
        # calculate and save if the file does not exist
        if not os.path.exists(os.path.join(os.getcwd(), self.RESULTS_FOLDER, "bank", self.PATH_BEGIN + "bank_profit")):
            # calculate the bank profit
            bank_profit = dfs_map["bank"]["test_df"].groupby('case_nr')['outcome'].first().sum()
            save_data(bank_profit, os.path.join(os.getcwd(), self.RESULTS_FOLDER, "bank", self.PATH_BEGIN + "bank_profit"))
        else:
            bank_profit = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, "bank", self.PATH_BEGIN + "bank_profit"))

        # RANDOM POLICY
        # calculate and save if the file does not exist
        if not os.path.exists(os.path.join(os.getcwd(), self.RESULTS_FOLDER, "random", self.PATH_BEGIN + str(self.iter) + "_random_profit")):
            # calculate the random profit
            random_profit = dfs_map["random_" + str(self.iter)]["test_df"].groupby('case_nr')['outcome'].first().sum()
            random_uplift = (random_profit - bank_profit) / abs(bank_profit) * 100 if bank_profit != 0 else 0
            save_data(random_profit, os.path.join(os.getcwd(), self.RESULTS_FOLDER, "random", self.PATH_BEGIN + str(self.iter) + "_random_profit"))
            save_data(random_uplift, os.path.join(os.getcwd(), self.RESULTS_FOLDER, "random", self.PATH_BEGIN + str(self.iter) + "_random_uplift"))
        else:
            random_profit = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, "random", self.PATH_BEGIN + str(self.iter) + "_random_profit"))
            random_uplift = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, "random", self.PATH_BEGIN + str(self.iter) + "_random_uplift"))
        
        # OPTIMAL POLICY
        if not os.path.exists(os.path.join(os.getcwd(), self.RESULTS_FOLDER, "optimal", self.PATH_BEGIN + "optimal_profit")):
            # calculate the optimal profit
            optimal_profit = dfs_map["optimal"]["test_df"].groupby('case_nr')['outcome'].first().sum()
            optimal_uplift = (optimal_profit - bank_profit) / abs(bank_profit) * 100 if bank_profit != 0 else 0
            save_data(optimal_profit, os.path.join(os.getcwd(), self.RESULTS_FOLDER, "optimal", self.PATH_BEGIN + "optimal_profit"))
            save_data(optimal_uplift, os.path.join(os.getcwd(), self.RESULTS_FOLDER, "optimal", self.PATH_BEGIN + "optimal_uplift"))
        else:
            optimal_profit = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, "optimal", self.PATH_BEGIN + "optimal_profit"))
            optimal_uplift = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, "optimal", self.PATH_BEGIN + "optimal_uplift"))

        return bank_profit, random_profit, random_uplift, optimal_profit, optimal_uplift

    def get_actions_recommended_current_stage(self, stage, target, index_to_select, model_params, collated_prep, case_nrs, to_add_cross_fitting):
        
        print(f"        Stage: {stage}, Target: {target}")

        model_to_load = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, self.method, "training", self.PATH_BEGIN + self.method + "_" + target + "_" + model_params["model_specific"] + "_" + str(index_to_select) + "_" + to_add_cross_fitting + str(self.iter) + "_training_model"), is_state_dict=(model_params["model_category"] == "dl" and "-S-" in model_params["method"]))

        model_evaluator = ModelEval(data=collated_prep, model_params=model_params, model_to_load=model_to_load, stage=stage)

        action_df = model_evaluator.eval()

        # === Ensure all case_nrs have a decision ===
        action_map = dict(zip(action_df["case_nr"], action_df["action"]))
        # Fill missing case_nrs with action 0
        full_action_list = [action_map.get(case_nr, 0) for case_nr in case_nrs]
        action_df = pd.DataFrame({"case_nr": case_nrs, "action": full_action_list})

        return action_df
    
    def row_to_tuple(self, row, action_cols):
        return tuple(row[c] for c in action_cols)
    
    def get_data_given_prev_actions(self, stage, preps_maps, model_params, target, index_to_select, prev_decisions):
        if stage == 0:
            key = str(tuple(0 for _ in range(self.args.n_stages)))
            collated_prep = preps_maps[key][model_params["encoding"]][0]
        else:
            encoding = self.model_params_list_of_dicts[index_to_select][target]["encoding"]
            selected_data = []

            dfs = []
            for s in range(stage):
                df = prev_decisions[s].copy()
                df = df[["case_nr", "action"]].rename(columns={"action": f"action_s{s}"})
                dfs.append(df)
            merged = reduce(lambda left, right: pd.merge(left, right, on="case_nr", how="inner"), dfs)
            action_cols = [f"action_s{s}" for s in range(stage)]
            merged["action_combo"] = merged.apply(lambda row: self.row_to_tuple(row=row, action_cols=action_cols), axis=1)
            combo_to_case_nrs: Dict[Tuple, List] = (merged.groupby("action_combo")["case_nr"].apply(list).to_dict())

            # NEW
            for combo, case_nrs in combo_to_case_nrs.items():
                str_items = ", ".join(repr(x) for x in combo + (0,)*(self.args.n_stages - len(combo)))
                key = f"({str_items})"

                try:
                    data_dict = preps_maps[key][encoding][index_to_select]
                    if model_params.get("method") == "kmeans_q":
                        # Fast vectorized filtering using pandas
                        filtered_rows = data_dict[data_dict["case_nr"].isin(case_nrs)]
                        selected_data.extend(filtered_rows.to_dict(orient="records"))
                    else:
                        # Torch tensor logic
                        case_nr_tensor = data_dict["case_nr"]
                        if not isinstance(case_nr_tensor, torch.Tensor):
                            case_nr_tensor = torch.tensor(case_nr_tensor.values if hasattr(case_nr_tensor, "values") else case_nr_tensor)

                        for case_nr in case_nrs:
                            try:
                                matching_indices = (case_nr_tensor == case_nr).nonzero(as_tuple=True)[0]
                                if len(matching_indices) > 0:
                                    i = matching_indices[0].item()
                                    selected_entry = {
                                        k: (v[i] if v is not None else None)
                                        for k, v in data_dict.items()
                                    }
                                    selected_data.append(selected_entry)
                            except (IndexError, RuntimeError, KeyError):
                                pass
                except KeyError:
                    pass

            if self.method == "kmeans_q":
                # just make a df again
                collated_prep = pd.DataFrame(selected_data)
                print('')
            else:
                # Collate into dict of tensors
                all_keys = selected_data[0].keys()  # Get all keys from one selected entry
                collated_prep = {}
                for key in all_keys:
                    values = [d[key] for d in selected_data]
                    if any(v is not None for v in values):
                        collated_prep[key] = torch.stack([v for v in values if v is not None])
                    else:
                        collated_prep[key] = None
        
        return collated_prep

    def calculate_proft_of_decisions(self, decisions, dfs_map):
        # Step 1: Merge all decision DataFrames on 'case_nr'
        all_decisions_df = None
        for decision_point, df in decisions.items():
            df = df[['case_nr', 'action']].rename(columns={'action': f'action_{decision_point}'})
            if all_decisions_df is None:
                all_decisions_df = df
            else:
                all_decisions_df = pd.merge(all_decisions_df, df, on='case_nr')

        # Step 2: Create action sequence tuple for each case
        decision_keys = sorted(decisions.keys())  # Ensure consistent order
        all_decisions_df['action_sequence'] = all_decisions_df.apply(
            lambda row: tuple(row[f'action_{k}'] for k in decision_keys), axis=1
        )

        # Step 3: Look up and collect matching rows from dfs_map
        final_dfs = []
        for action_sequence, group in all_decisions_df.groupby('action_sequence'):
            key = str(action_sequence)  # Convert tuple to string key for dfs_map
            if key in dfs_map:
                matching_cases = group['case_nr']
                filtered_df = dfs_map[key]["test_df"][dfs_map[key]["test_df"]['case_nr'].isin(matching_cases)]
                final_dfs.append(filtered_df)

        # Step 4: Concatenate all results
        final_df = pd.concat(final_dfs, ignore_index=True)

        # Sum up the whole df to get one float (so get one value of column 'outcome' for each case_nr in the final_df)
        profit = final_df.groupby('case_nr')['outcome'].first().sum()

        return profit, final_df