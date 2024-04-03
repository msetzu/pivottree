from RuleTree import *
from PivotTree import *


import pandas as pd
import numpy as np
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


import sys


# Default values
data_set_name = ''
mode_of_evaluation = 'validation'  # or 'testing'
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

print(training_folder + data_set_name)

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

    
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier

from alibi.prototypes import ProtoSelect
from alibi.utils.kernel import EuclideanDistance
from alibi.prototypes.protoselect import cv_protoselect_euclidean




#possible_configurations = ([(i, np.percentile(X_train_inner_pairwise_matrix, i)) for i in range(2,42,2)])[::-1]
possible_configurations = [(i / 100,i / 100 )  for i in range(2,42,2)]
possible_configurations



def select_epsilon_ball(X_train_inner, y_train_inner, X_validation, y_validation , params_list, num_prototypes = None):


    if num_prototypes == None:
        num_prototypes = len(X_train_inner) * 2
    grid_size = 2
    quantiles = params_list

    # search for the best epsilon-radius value

    start_time = time.process_time()
    cv = cv_protoselect_euclidean(trainset=(X_train_inner, y_train_inner),
                                valset=(X_validation, y_validation),
                                num_prototypes=num_prototypes,
                                quantiles=quantiles,
                                grid_size=grid_size)
    

    summariser = ProtoSelect(kernel_distance=EuclideanDistance(),
                          eps=cv['best_eps'],
                          )
    
    summariser = summariser.fit(X=X_train_inner, y=y_train_inner)
    summary = summariser.summarise(num_prototypes=num_prototypes)

    end_time = time.process_time()

    time_selection = end_time - start_time
      
    X_pivots = summary.data['prototypes']
    y_pivots = summary.data['prototype_labels']


    return X_pivots, y_pivots, time_selection


def evaluate_epsilon(X_train, y_train, X_validation, y_validation, X_pivots, y_pivots, configuration, class_wise = False):


  results_list = []

  #KNN PERFORMANCE ON ORIGINAL SPACE

  for neigh in [1,3,5]:

      if len(y_pivots) < neigh:
        continue

      num_leaves = 0.0

      knn = KNeighborsClassifier(n_neighbors=neigh, metric = metric)

      time_fit = 0.0
      start_time = time.process_time()
      knn.fit(X_pivots, y_pivots)
      end_time = time.process_time()
      time_fit = end_time - start_time

      time_predict = 0.0
      start_time = time.process_time()
      predictions = knn.predict(X_validation)
      end_time = time.process_time()
      time_predict = end_time - start_time

      KNN_score_dict = compute_score(y_validation, predictions)

      KNN_score_dict['configuration'] = 'EPSILON_KNN' + str(neigh) + '_original' + '_' + str(configuration)
      KNN_score_dict['total_pivots_num'] = len(y_pivots)
      KNN_score_dict['random_state'] = random_state
      KNN_score_dict['num_leaves'] = num_leaves
      KNN_score_dict['time_selection'] = time_selection
      KNN_score_dict['time_partition'] = time_partition
      KNN_score_dict['time_fit'] = time_fit
      KNN_score_dict['time_predict'] = time_predict
      KNN_score_dict['time_pairwise_training'] = 0.0
      KNN_score_dict['time_pairwise_validation'] = 0.0
      KNN_score_dict['data_set_name'] = data_set_name


      if class_wise:
        KNN_score_dict['configuration'] = 'CLASS_WISE_' + KNN_score_dict['configuration']

      results_list += [KNN_score_dict]

   #KNN PERFORMANCE ON SPACE_TRANSFORMED

  X_train_reduced, time_pairwise_training = pariwise_computation(X_train, X_pivots, metric) #reduce training set
  X_validation_reduced, time_pairwise_validation = pariwise_computation(X_validation, X_pivots, metric) #reduce validation set

  for neigh in [1,3,5]:

      if len(y_pivots) < neigh:
        continue

      

      #if i extract less prottoype then config for KNN, i skip it
          
      num_leaves = 0.0

      knn = KNeighborsClassifier(n_neighbors=neigh, metric = metric)

      time_fit = 0.0
      start_time = time.process_time()
      knn.fit(X_train_reduced, y_train)
      end_time = time.process_time()
      time_fit = end_time - start_time

      time_predict = 0.0
      start_time = time.process_time()
      predictions = knn.predict(X_validation_reduced)
      end_time = time.process_time()
      time_predict = end_time - start_time

      KNN_score_dict = compute_score(y_validation, predictions)

      KNN_score_dict['configuration'] = 'EPSILON_KNN' + str(neigh) + '_mapped' + '_' + str(configuration)
      KNN_score_dict['total_pivots_num'] = len(y_pivots)
      KNN_score_dict['random_state'] = random_state
      KNN_score_dict['num_leaves'] = num_leaves
      KNN_score_dict['time_selection'] = time_selection
      KNN_score_dict['time_partition'] = time_partition
      KNN_score_dict['time_fit'] = time_fit
      KNN_score_dict['time_predict'] = time_predict
      KNN_score_dict['time_pairwise_training'] = time_pairwise_training
      KNN_score_dict['time_pairwise_validation'] = time_pairwise_validation
      KNN_score_dict['data_set_name'] = data_set_name


      if class_wise:
        KNN_score_dict['configuration'] = 'CLASS_WISE_' + KNN_score_dict['configuration']

      results_list += [KNN_score_dict]


   #DT PERFORMANCE ON SPACE TRANSFORMED

  for depth in [4]:
  
        if len(y_pivots) < 1:
           continue

        dt = DecisionTreeClassifier(max_depth = depth, min_samples_leaf = 3, min_samples_split = 5, random_state = 0)

        time_fit = 0.0
        start_time = time.process_time()
        dt.fit(X_train_reduced, y_train)
        end_time = time.process_time()
        time_fit = end_time - start_time

        time_predict = 0.0
        start_time = time.process_time()
        predictions = dt.predict(X_validation_reduced)
        end_time = time.process_time()
        time_predict = end_time - start_time

        num_leaves = dt.get_n_leaves()

        DT_score_dict = compute_score(y_validation, predictions)

        DT_score_dict['configuration'] = 'EPSILON_DT' + str(depth) + '_mapped' + '_' + str(configuration)
        DT_score_dict['total_pivots_num'] = len(y_pivots)
        DT_score_dict['random_state'] = random_state
        DT_score_dict['num_leaves'] = num_leaves
        DT_score_dict['time_selection'] = time_selection
        DT_score_dict['time_partition'] = time_partition
        DT_score_dict['time_fit'] = time_fit
        DT_score_dict['time_predict'] = time_predict
        DT_score_dict['time_pairwise_training'] = time_pairwise_training
        DT_score_dict['time_pairwise_validation'] = time_pairwise_validation
        DT_score_dict['data_set_name'] = data_set_name


        if class_wise:
          DT_score_dict['configuration'] = 'CLASS_WISE_' + DT_score_dict['configuration']


        results_list += [DT_score_dict]
  
   
  return results_list


#for each random state, fit a pivot tree on the validation set and select the prototypes

result_list = []
result_list_limited = []

for random_state in range(5):
    idxs = np.array([x for x in range(len(X_train_inner))])
    np.random.seed(random_state)
    np.random.shuffle(idxs)

    X_train_inner = X_train_inner[idxs]
    y_train_inner = y_train_inner[idxs]

    for params_list in possible_configurations:

        X_pivots, y_pivots, time_selection = select_epsilon_ball(X_train_inner, y_train_inner, X_validation, y_validation, params_list)

        if len(X_pivots) <= 1:
            continue

        result_list = result_list + evaluate_epsilon(X_train_inner, y_train_inner, X_validation, y_validation, X_pivots, y_pivots, params_list)

        X_pivots_limited, y_pivots_limited, time_selection_limited = select_epsilon_ball(X_train_inner, y_train_inner, X_validation, y_validation, params_list, num_prototypes=20)

        if len(X_pivots_limited) <= 1:
            continue

        result_list_limited = result_list_limited + evaluate_epsilon(X_train_inner, y_train_inner, X_validation, y_validation, X_pivots_limited, y_pivots_limited, params_list)

    

    
    



unique_configs = set([x['configuration'] for x in result_list])

dictionaries_result = []

for config in unique_configs:
  current_results = [x for x in result_list if x['configuration'] == config]


  dictionaries_result.append(compute_average_and_std(current_results))


avg_dict = pd.DataFrame(dictionaries_result)
avg_dict.to_csv(data_set_name + '_EPSILON_' + mode_of_evaluation + '.csv', index = False)

avg_dict

print(avg_dict.iloc[0])




unique_configs = set([x['configuration'] for x in result_list_limited])

dictionaries_result = []

for config in unique_configs:
  current_results = [x for x in result_list_limited if x['configuration'] == config]
  #print(len(current_results))
  #print(current_results)

  dictionaries_result.append(compute_average_and_std(current_results))


avg_dict = pd.DataFrame(dictionaries_result)
avg_dict.to_csv(data_set_name + '_EPSILON_' + mode_of_evaluation + '_limited_' + '.csv', index = False)

avg_dict

print(avg_dict.iloc[0])













