from src.utils.mini_tools import split_raw_data, save_data
from src.utils.prep_tools.tensor_prep import TensorPreprocessor
from src.utils.prep_tools.agg_prep import AggPreprocessor
from src.utils.prep_tools.kmeans_prep import KMeansPreprocessor
import os

class ProcessPreprocessor():
    def __init__(self, args, raw_data, DATASET_PARAMS_LIST):
        to_add = ''
        if args.dataset == 'bpic17':
            conf_suffix = "_case" if args.confounding_type == "case" else ""
            to_add = os.path.join("bpic17" + conf_suffix, str(args.n_stages))
        self.DATA_FOLDER = os.path.join("data", to_add, str(args.train_size), str(int(100 * args.delta)))
        self.PATH_BEGIN = str(args.train_size) + "_" + str(int(100*args.delta)) + "_"

        self.args = args
        self.raw_data = raw_data
        self.DATASET_PARAMS_LIST = DATASET_PARAMS_LIST

        if "elapsed_time" in self.raw_data.columns:
            self.data_ordered = raw_data.sort_values(by=["case_nr", "elapsed_time"])
        elif "event_nr" in self.raw_data.columns:
            self.data_ordered = raw_data.sort_values(by=["case_nr", "event_nr"])
        else:
            self.data_ordered = raw_data.sort_values(by=["case_nr"])

        max_process_len = self.data_ordered.groupby(["case_nr"]).size().max() - 1 # Minus one because the last activity is cancel or accept application, not useful for our inference
        missing_value = -100

        self.add_properties = {
            "max_process_len": max_process_len,
            "missing_value": missing_value
        }

    
    def preprocess(self, encoding, prep_utils_list=None, eval=False, action_combo=""):
        
        if eval:
            self.PREP_PARAMS =  {"infer_prop": 1.0, "train_prop": 0.0, "filter_useless_cols": True}
            self.DATA_FOLDER_TOTAL = os.path.join(self.DATA_FOLDER, "eval")
            self.PATH_END_DICT = {"infer": "_preprocessed_data_eval", "utils": "_preprocessed_utils_eval"}
        else:
            self.PREP_PARAMS =  {"infer_prop": 0.2, "train_prop": 0.8, "filter_useless_cols": True}
            self.DATA_FOLDER_TOTAL = os.path.join(self.DATA_FOLDER, "training_and_tuning")
            self.PATH_END_DICT = {"train": "_preprocessed_data_train", "infer": "_preprocessed_data_infer", "utils": "_preprocessed_utils"}

        self.data_train, self.data_infer = split_raw_data(data=self.data_ordered, infer_prop=self.PREP_PARAMS["infer_prop"])
        
        if "kmeans_q" in self.args.methods:
            n_stages = 1
        else: n_stages = self.args.n_stages

        self.data_train_prep_list = []
        self.data_infer_prep_list = []
        self.prep_utils_list = []
        for stage in range(n_stages):
            DATASET_PARAMS = self.DATASET_PARAMS_LIST[stage]
            nr_treatment_columns = DATASET_PARAMS["intervention_info"]["action_width"] if DATASET_PARAMS["intervention_info"]["action_width"] > 2 else 1
            self.add_properties["nr_treatment_columns"] = nr_treatment_columns
            self.add_properties["stage"] = stage

            if encoding == "tensor":
                self.preprocessor = TensorPreprocessor(self.data_train, self.data_infer, self.PREP_PARAMS, DATASET_PARAMS, self.add_properties, prep_utils=prep_utils_list[stage] if prep_utils_list is not None else None)
            elif encoding == "agg":
                self.preprocessor = AggPreprocessor(self.data_train, self.data_infer, self.PREP_PARAMS, DATASET_PARAMS, self.add_properties, prep_utils=prep_utils_list[stage] if prep_utils_list is not None else None)
            elif encoding == "kmeans":
                # NOTE: returns data_infer as None, because it is not used in the KMeansPreprocessor
                self.preprocessor = KMeansPreprocessor(self.data_train, self.data_infer, self.PREP_PARAMS, DATASET_PARAMS, prep_utils=prep_utils_list[stage] if prep_utils_list is not None else None, args=self.args)

            data_train_prep, data_infer_prep, prep_utils = self.preprocessor.preprocess()
            self.data_train_prep_list.append(data_train_prep)
            self.data_infer_prep_list.append(data_infer_prep)
            self.prep_utils_list.append(prep_utils)

            # Save
            if not eval:
                save_data(data_train_prep, os.path.join(os.getcwd(), self.DATA_FOLDER_TOTAL, self.PATH_BEGIN + encoding + "_" + str(stage) + "_" + action_combo + self.PATH_END_DICT["train"]))
            save_data(data_infer_prep, os.path.join(os.getcwd(), self.DATA_FOLDER_TOTAL, self.PATH_BEGIN + encoding + "_" + str(stage) + "_" + action_combo + self.PATH_END_DICT["infer"]))
            save_data(prep_utils, os.path.join(os.getcwd(), self.DATA_FOLDER_TOTAL, self.PATH_BEGIN + encoding + "_" + str(stage) + "_" + action_combo + self.PATH_END_DICT["utils"]))

        return self.data_train_prep_list, self.data_infer_prep_list, self.prep_utils_list