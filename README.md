# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about bank marketing campaigns with information about the customers that were contacted. The goal of the project is to predict whether a client will subscribe a term deposit. We seek to predict whether a given client with a set of previously known attributes would or would not subscribe a term deposit.

The best performing model was a VotingEnsemble with an accuracy of 0.9170.

## Scikit-learn Pipeline
The pipeline architecture is as follows:

* Data loaded from TabularDatasetFactory
* Dataset is further cleaned using a function called clean_data
* One Hot Encoding has been used for the categorical columns
* Dataset has been splitted into train and test sets
* Built a Logistic Regression model
* Hyperparameters has been tuned using Hyperdrive
* Selected the best model
* Saved the model 

**Benefits of the parameter sampler chosen**
The project used Random Sampling as it supports early termination of low-performance runs. In random sampling, hyperparameter values are randomly selected from the defined search space with two hyperparameters '--C' (Reqularization Strength) and '--max_iter' (Maximum iterations to converge).
Random sampling search is also faster, allows more coverage of the search space and parameter values are chosen from a set of discrete values or a distribution over a continuous range.

**The benefits of the early stopping policy chosen**
The Bandit policy was chosen because it stops a run if the target performance metric underperforms the best run so far by a specified margin. It ensures that we don't keep running the experiment running for too long and end up wasting resources and time looking for the optimal parameter. It is based on slack criteria and a frequency and delay interval for evaluation. Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated.

## AutoML
The best performing model for AutoML in this instance was VotingEnsemble model which combines multiple models to produce a better result compared to a single model. It gave as an accuracy score of 0.9170 which was slightly better than the score achieved using HyperDrive.

## Pipeline comparison
Though both the models used automated machine learning, a difference in the accuracies was visible, with model trained using AutoML gave slightly better results. The AutoML model gave best accuracy of 0.9170, while the model built using SKLearn and HyperDrive gave a slightly lower score of 0.9144.

Compared to HyperDrive, AutoML architecture is quite superior, which enables to training 'n' number of models efficiently.

The reason in accuracies might be due to the fact that that we used less number of iterations in AutoML run, which might give better results with more iterations. AutoML also provides a wide variety of models and preprocessing steps which are not carried out Hyperdrive. However, the difference was quite small.

## Future work
For future experiments, we can try more hyperparameters for the HyperDrive model, test different sampling methods as well as have a larger search space to maximize the search. We can also try different models and see if we get a better accuracy and train a more robust model for inferencing.

For AutoML, we can try implementing explicit model complexity limitations to prevent over-fitting. We can also test out different parameter values such as number of folds for Cross Validation. IN addition we can try working with raw data only and passing it to AutoML to see how it handles it and if it will affect the chosen model and the model accuracy. Reducing over-fitting is an important task that may improve model accuracy. If a model is over-fitting, it might have a high accuracy with training data, but will fail when performing inferencing on test data.

Lastly, the dataset seems to be imbalanced. This means that the accuracy is affected due to imbalanced classes in the input, because the input data is biased towards certain classes. To handle the imbalance we can try to change sampling technique or performance matrix. 
