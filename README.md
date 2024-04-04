### Dataset info

`datasets_info.md` contains info about datasets sources. The pre-processing employed for each dataset is specified in the relative paper section.


### Experiment replication
The following .py files are used to execute the respective experiments:
- `EPSILON_validation.py` is used to generate validation/testing results for epsilon-ball with ProtoSelect implemented in alibi library (https://docs.seldon.io/projects/alibi/en/stable/index.html)
- `RANDOM_validation.py` is used to generate validation/testing results with random or class-wise random selection
- `CENTROIDS_validation.py` is used to generate validation/testing results with k-Means and k-Medoids algorithms
- `STANDARD_validation.py` is used to generate validation/testing results with standard sklearn classifiers
- `PT_validation.py` is used to generate validation/testing results with PivotTree, both as a classifier and as a prototype selection tool

#### Notes on experiments
For all the experiments, a folder named `dataset_stz` must be avilable containing a folder `dataset_stz/splits`, respectively containing
- `dataset_stz/splits/training_sets`
- `dataset_stz/splits/validation_sets`
- `dataset_stz/splits/test_sets`

Each folder hosts the training, validation and testing splits from all the datasets taken into account. Therefore, given a dataset named `df_name.csv`, we will have in the respecitve folders `train_df_name.csv`, `validation_df_name.csv`, `test_df_name.csv`, where `df_name.csv` stands for the name of the full dataset.

The folder `dataset_stz` contains for each data set used `df_name.csv` this being the full dataset obtained combining train, validation and test splits.

Where an original train/test split was not available, the train/test split was obtained using `sklearn.model_selection.train_test_split` with a random state of 42 and a 70/30 partition from original datasets.

For further splits of the training set, the train/validation split was obtained using the same method with a random state of 42 and a 80/20 partition from original datasets.

#### Running experiments

For `PT_validation.py`, in order to execute a validation/testing run, the following commands can be used:

`python PT_validation datasets_stz/df_name.csv approximation mode_of_evaluation`, where 

- `df_name.csv` is the dataset name
- approximation can be True/False
- mode_of_evalution can be 'validation' or 'testing'

For instance, to get validation results on a dataset named  `tennisgpt_embed.csv` without using the approximated version of PivotTree, you can run:

`python PT_validation datasets_stz/tennisgpt_embed.csv False validation`

For all the other methods, it is sufficent to specify:

- `python METHOD_validation datasets_stz/df_name.csv mode_of_evaluation`

without considering the approximation parameter. METHOD_validation can be identified as one of the other '_validation.py' files in the repository.

For instance, to get validation results on a dataset named  `tennisgpt_embed.csv` considering k-Means and k-Medoids selection methods, you can run:

`python CENTROIDS_validation datasets_stz/tennisgpt_embed.csv validation`

#### Qualitative and sensitivity results
Info about qualtiative visualizations are available in the `Qualitative_Pivot_Tree.ipynb` file; Info about sensitivity plots are available in `Sensitivity_Pivot_Tree.ipynb` file.

#### Quantitative, Simplicity and CD plots
Quantitative results in terms of F1-score and number of pivots results on the test set, as well as the simplicity plots, were obtained using matplotlib visualization approaches and considering results from the model assessment phase.

For CD plot, autorank library was used (https://pypi.org/project/autorank/0.1.0/) 




