from RuleTree import *
from PivotTree import *

import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import product
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, roc_auc_score, balanced_accuracy_score, f1_score, accuracy_score, recall_score, precision_score

def generate_param_configurations(configurations):
    keys, values_list = zip(*configurations.items())
    param_combinations = list(product(*values_list))

    result_configurations = []
    for combination in param_combinations:
        config_dict = dict(zip(keys, combination))
        result_configurations.append(config_dict)

    return result_configurations


import sys

data_set_name = ''
mode_of_evaluation = 'validation'  # or 'testing'
approximation = True
metric = 'euclidean'
time_partition = 0.0


def parse_arguments(args):
    global data_set_name, approximation, mode_of_evaluation
    
    # Parsing command line arguments
    if len(args) >= 2:
        data_set_name = args[1]
    if len(args) >= 3:
        mode_of_evaluation = args[2].lower()

parse_arguments(sys.argv)

#get dataset name

data_set_name

data_set_name = (data_set_name.split('/')[1])

print(data_set_name)


if 'scaled' in data_set_name:
    training_folder = 'datasets_stz_standardized/splits/training_sets/train_'
    validation_folder = 'datasets_stz_standardized/splits/validation_sets/validation_'
    test_folder = 'datasets_stz_standardized/splits/test_sets/test_'
else:
    training_folder = 'datasets_stz/splits/training_sets/train_'
    validation_folder = 'datasets_stz/splits/validation_sets/validation_'
    test_folder = 'datasets_stz/splits/test_sets/test_'


ds_train = pd.read_csv(training_folder + data_set_name)
ds_validation = pd.read_csv(validation_folder + data_set_name)
ds_test = pd.read_csv(test_folder + data_set_name)

class_name = ds_train.columns[-1]


X_train_inner = np.array(ds_train.loc[:, ds_train.columns != class_name].values)
y_train_inner = np.array(ds_train[class_name])

X_validation = np.array(ds_validation.loc[:, ds_validation.columns != class_name].values)
y_validation = np.array(ds_validation[class_name])

X_test = np.array(ds_test.loc[:, ds_test.columns != class_name].values)
y_test = np.array(ds_test[class_name])


if mode_of_evaluation == 'testing':
    print('Merge train with validation for testing')
    X_train_inner = np.array([x for x in X_train_inner] + [x for x in X_validation])
    y_train_inner = np.array([x for x in y_train_inner] + [x for x in y_validation])

    X_validation = X_test
    y_validation = y_test

print('Train/Val/Test size: ', len(y_train_inner), len(y_validation), len(y_test))

if mode_of_evaluation not in ['testing', 'validation']:
    raise ValueError('Shoulde be testing or validation')
    


data_set_name = 'datasets_stz/' + data_set_name



configuration = ''


def compute_score(y_validation, predictions):
  balanced_accuracy_score_ = balanced_accuracy_score(y_validation, predictions)
  f1_score_ = f1_score(y_validation, predictions, average='macro')
  accuracy_score_ = accuracy_score(y_validation, predictions)
  precision_score_ = precision_score(y_validation, predictions, average='macro')
  recall_score_ = recall_score(y_validation, predictions, average='macro')

  score_dict = {
        'balanced_accuracy_score': balanced_accuracy_score_,
        'f1_score': f1_score_,
        'accuracy_score': accuracy_score_,
        'precision_score': precision_score_,
        'recall_score': recall_score_
    }

  return score_dict

def compute_average_and_std(data_list):
    # Define the keys to consider for metrics
    keys_to_consider = [
        'accuracy_score',
        'balanced_accuracy_score',
        'f1_score',
        'precision_score',
        'recall_score',
        'time_partition',
        'time_selection',
        'time_pairwise_training',
        'time_pairwise_validation',
        'time_fit',
        'time_predict',
        'total_pivots_num',  # Include total_pivots_num in the keys
        'num_leaves',
    ]

    # Define the keys to keep constant
    keys_constant = [
        'configuration',
        'data_set_name',

    ]

    # Initialize a dictionary to store the computed values
    result_dict = {}

    # Include constant keys in the result
    for key in keys_constant:
        result_dict[key] = data_list[0][key]  # Assuming the constant values are the same for all entries

    # Extract the relevant metrics into separate lists for each key
    key_lists = {key: [entry[key] for entry in data_list] for key in keys_to_consider}

    # Compute the average and standard deviation for each key
    for key, values in key_lists.items():
        average_key = np.mean(values)
        std_key = np.std(values)
        result_dict[f'average_{key}'] = average_key
        result_dict[f'std_{key}'] = std_key

    return result_dict


def matching_entries(data, configuration_classifier_name):
    configuration = configuration_classifier_name[0]
    classifier_name = configuration_classifier_name[1]
    matching_entries = [entry for entry in data if entry['configuration'] == configuration and entry['classifier_name'] == classifier_name]
    if not matching_entries:
        print(f"No matching entries found for configuration: {configuration} and classifier: {classifier_name}")
        return None
    return matching_entries


def evaluate_base_classifiers(X_train, y_train, X_validation, y_validation):
    results_list = []

    for neigh in [1, 3, 5]:
        knn = KNeighborsClassifier(n_neighbors=neigh, metric=metric)

        time_fit = 0.0
        start_time = time.process_time()
        knn.fit(X_train, y_train)
        end_time = time.process_time()
        time_fit = end_time - start_time

        time_predict = 0.0
        start_time = time.process_time()
        predictions = knn.predict(X_validation)
        end_time = time.process_time()
        time_predict = end_time - start_time

        KNN_score_dict = compute_score(y_validation, predictions)

        KNN_score_dict['configuration'] = 'STNDRKNN_' + str(neigh) + '_original' + '' + str(configuration)
        KNN_score_dict['total_pivots_num'] = 0
        KNN_score_dict['random_state'] = random_state
        KNN_score_dict['num_leaves'] = 0.0
        KNN_score_dict['time_selection'] = 0.0
        KNN_score_dict['time_partition'] = 0.0
        KNN_score_dict['time_fit'] = time_fit
        KNN_score_dict['time_predict'] = time_predict
        KNN_score_dict['time_pairwise_training'] = 0.0
        KNN_score_dict['time_pairwise_validation'] = 0.0
        KNN_score_dict['data_set_name'] = data_set_name

        results_list += [KNN_score_dict]

    for depth in [4]:
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=3, min_samples_split=5, random_state=random_state)

        time_fit = 0.0
        start_time = time.process_time()
        dt.fit(X_train, y_train)
        end_time = time.process_time()
        time_fit = end_time - start_time

        time_predict = 0.0
        start_time = time.process_time()
        predictions = dt.predict(X_validation)
        end_time = time.process_time()
        time_predict = end_time - start_time

        num_leaves = dt.get_n_leaves()

        DT_score_dict = compute_score(y_validation, predictions)

        DT_score_dict['configuration'] = 'STNDRDT_' + str(depth) + '_mapped' + '' + str(configuration)
        DT_score_dict['total_pivots_num'] = 0.0
        DT_score_dict['random_state'] = random_state
        DT_score_dict['num_leaves'] = num_leaves
        DT_score_dict['time_selection'] = 0.0
        DT_score_dict['time_partition'] = 0.0
        DT_score_dict['time_fit'] = time_fit
        DT_score_dict['time_predict'] = time_predict
        DT_score_dict['time_pairwise_training'] = 0.0
        DT_score_dict['time_pairwise_validation'] = 0.0
        DT_score_dict['data_set_name'] = data_set_name

        results_list += [DT_score_dict]

    return results_list

#for each random state, fit a pivot tree on the validation set and select the prototypes

result_list = []

for random_state in range(5):

  for params_list in ['']:

    
    result_list = result_list + evaluate_base_classifiers(X_train_inner, y_train_inner, X_validation, y_validation)


unique_configs = set([x['configuration'] for x in result_list])


dictionaries_result = []

for config in unique_configs:
  current_results = [x for x in result_list if x['configuration'] == config]
  #print(len(current_results))
  #print(current_results)

  dictionaries_result.append(compute_average_and_std(current_results))


avg_dict = pd.DataFrame(dictionaries_result)
avg_dict.to_csv(data_set_name + '_STNDR_' + mode_of_evaluation + '.csv', index = False)

avg_dict
