
import heapq
import operator
import pickle

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from river import (
    stream, compose, metrics, ensemble
)

from river import linear_model as lm
from river import preprocessing as pp


# Seed should also come from config.
# But keeping it here for this exercise
SEED: int = 42
np.random.seed(SEED)


class OnlineTraining:
    def __init__(
            self,
            predictive_df: pd.DataFrame,
            n_tweets_to_show: int = 20,
            test_size: float = 0.3

    ):
        self.predictive_df = predictive_df
        self.n_tweets_to_show = n_tweets_to_show
        self.test_size = test_size
        # innit variables
        self.reset_model: bool = False
        self._model, self._metric = self._online_model
        self._model_trained = False
        self.recommended: dict = dict()
        self._metric_scores = None
        self._confusion_matrix = None

        self._fake_user_input = self._return_fake_user_input

    def _preprocess_training_data(
            self,
    ):
        """This method should also include all the pre-filtering
        and pre-processing pipelines which would allow us to gain
        efficiency and perform better rankings.
        For example, heuristics and rule based algorithms can be added here,
        as well as pre-filters to avoid displaying un-interesting tweets.

        """

        processed_df = self.predictive_df
        # encode label (target)
        processed_df['label'] = processed_df.label.apply(
                lambda x: 1 if x == 'accepted' else 0
        )

        # Adding an ID for now with simple numpy
        # we would assume this ID would be generated in the source
        processed_df["tweet_id"] = np.random.randint(
            low=111111, high=999999, size=len(processed_df)
        )

    def _build_train_test_data(
        self
    ):
        # making sure the prediction data is ready
        self._preprocess_training_data()
        # Build training Set
        X = self.predictive_df.drop('label', axis=1)
        y = self.predictive_df.label
        # Split train test
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, stratify=y, random_state=SEED
        )
        return X_train, X_test, y_train, y_test

    @property
    def _online_model(
            self
    ):
        """Builds the Model to be used. This method should be formalized
        in an experiment-based model construction exercise. Which would
        include multiple ML good practices as, feature selection,
        class balancing, outlier removal/imputation, etc.

        :return: Model preprocessing and training pipeline
        :rtype: river.compise.Pipeline
        """
        model = compose.Pipeline(
            ('scale', pp.StandardScaler()),
            (
                'stack', ensemble.StackingClassifier(
                    [
                        lm.LogisticRegression(),
                        lm.PAClassifier(mode=1, C=0.01),
                        lm.PAClassifier(mode=2, C=0.01),
                    ],
                    meta_classifier=lm.LogisticRegression()
                )
            )
        )
        metric = metrics.ROCAUC()
        return model, metric

    @_online_model.setter
    def _online_model(
        self, new_model, new_metric
    ):
        if (
            (isinstance(new_model, compose.Pipeline)) &
            (isinstance(new_metric, metrics))
        ):
            self._model, self._metric = new_model, new_metric
        else:
            print(
                 "New Model has to be type river.compose.pipeline.Pipeline"
                 " and New Metric has to be type river.metrics"
            )

    @staticmethod
    def plot_metrics(scores: list, confusion_matrix):
        print('Confusion Matrix')
        print(confusion_matrix)
        print('------------------------------')
        # plot performance score metrics
        iters = range(len(scores))
        ax = sns.lineplot(x=iters, y=scores)
        ax.set(xlabel='num_iters', ylabel='ROCAUC score')
        plt.show()

    def train_online_model(
            self,
            plot_metrics: bool = True
    ):
        """This is where the model is trained with the training set.
        Still uses the streaming framework which is then incorporated
        into other methods for prediction in this class

        :param plot_metrics: Plot performance metrics, defaults to True
        :type plot_metrics: bool, optional
        """
        # Split Test Train
        X_train, X_test, y_train, y_test = self._build_train_test_data()
        # Innit variables
        recommended = dict()
        counter = 0  # for now ID will simply be df.index TODO FIXME
        min_value_in_dict = 0
        metric_scores = list()
        cm = metrics.ConfusionMatrix()
        # Start Online Stream Training
        for xi, yi in stream.iter_pandas(
            X_train, y_train,
            shuffle=False, seed=SEED
        ):
            id = xi.get('tweet_id')
            xi.pop('tweet_id')
            # Test the current model on the new "unobserved" sample
            yi_pred = self._model.predict_proba_one(xi)
            # Update the running metric with the prediction
            # and ground truth value
            self._metric.update(yi, yi_pred)
            metric_scores.append(self._metric.get())
            # also update the confusion matrix
            cm.update(yi, round(yi_pred.get(True)))

            # Train the model with the new sample
            self._model.learn_one(xi, yi)

            if yi_pred.get(True) > min_value_in_dict:
                recommended[id] = yi_pred.get(True)
                # This should use a better and faster sorting. TODO
                keys = heapq.nlargest(
                    self.n_tweets_to_show, recommended, key=recommended.get
                )
                recommended = {
                    k: v for k, v in recommended.items() if k in keys
                }
                recommended = dict(
                    sorted(
                        recommended.items(),
                        key=operator.itemgetter(1),
                        reverse=True
                    )
                )
                min_value_in_dict = min(recommended.values())

            counter += 1

        if plot_metrics:
            print(f'ROC AUC: {self._metric}')
            OnlineTraining.plot_metrics(
                metric_scores,
                cm
            )
        # Setter for variables after train
        self.recommended = recommended
        self._metric_scores = metric_scores
        self._model_trained = True
        self._confusion_matrix = cm
        # Evaluate in the Test Set
        self._evaluate_training(
            X_test,
            y_test,
            plot_metrics=plot_metrics
        )

    def _evaluate_training(
            self,
            X_test: pd.DataFrame,
            y_test: pd.DataFrame,
            plot_metrics: bool = True
    ):
        """Evaluates the results from the training using a test set.

        :param X_test: Test Data (predictive variables)
        :type X_test: pd.DataFrame
        :param y_test: target varible in the test data
        :type y_test: pd.DataFrame
        :param plot_metrics: Plot performance metrics, defaults to True
        :type plot_metrics: bool, optional
        :raises NotImplementedError: If the model hasn't been traied yet.
        """
        if not self._model_trained:
            raise NotImplementedError(
                'Model has not been trained yet'
            )

        recommended = self.recommended
        min_value_in_dict = min(recommended.values())
        for xi, yi in stream.iter_pandas(
            X_test, y_test,
            shuffle=False, seed=SEED
        ):

            id = xi.get('tweet_id')
            xi.pop('tweet_id')
            # Test the current model on the new "unobserved" sample
            yi_pred = self._model.predict_proba_one(xi)

            # Update the running metric with the prediction
            # and ground truth value
            self._metric.update(yi, yi_pred)
            self._metric_scores.append(self._metric.get())
            # also update the confusion matrix
            self._confusion_matrix.update(yi, round(yi_pred.get(True)))

            # Train the model with the new sample
            self._model.learn_one(xi, yi)

            if yi_pred.get(True) > min_value_in_dict:
                recommended[id] = yi_pred.get(True)
                # This should use a better and faster sorting. TODO
                keys = heapq.nlargest(
                    self.n_tweets_to_show, recommended, key=recommended.get
                )
                recommended = {
                    k: v for k, v in recommended.items() if k in keys
                }
                recommended = dict(
                    sorted(
                        recommended.items(),
                        key=operator.itemgetter(1),
                        reverse=True
                    )
                )
                min_value_in_dict = min(recommended.values())

        self.recommended = recommended
        if plot_metrics:
            print(f'ROC AUC After Test Set learning: {self._metric}')
            OnlineTraining.plot_metrics(
                self._metric_scores,
                self._confusion_matrix
            )

    @staticmethod
    def _process_new_entry(
            entry: pd.Series
    ):
        """Preprocess an entry for prediction

        :param entry: Entry, must follow test set formatting.
        :type entry: pd.Series
        :return: processed entry input for modelling
        :rtype: dict
        """

        entry["tweet_id"] = np.random.randint(
                    low=111111, high=999999, size=1
                )[0]
        return entry.to_dict()

    def new_entry(
            self,
            tweet_input: dict,
            plot_metrics: bool = True

    ):
        """his is equivalent to the predict method in many ML libraries.
        However, it performs a single prediction for a given
        entry in the stream.

        :param tweet_input: Tweet input to predict, processed by
        OnlineTraining._process_new_entry
        :type tweet_input: dict
        :param plot_metrics: Plot performance metrics, defaults to True
        :type plot_metrics: bool, optional
        :raises NotImplementedError: If model hasn't been trained yet
        """
        if not self._model_trained:
            raise NotImplementedError(
                'Model has not been trained yet'
            )
        tweet_input = OnlineTraining._process_new_entry(tweet_input)
        # Get the id
        id = tweet_input.get('tweet_id')
        tweet_input.pop('tweet_id')

        # Make predictions
        yi_pred = self._model.predict_proba_one(
            tweet_input
        )

        yi = float(self._fake_user_input)  # make sure is float.
        print(f"Tweet ID {id}")
        print(yi_pred)
        self._metric.update(yi, yi_pred)
        self._metric_scores.append(self._metric.get())
        # also update the confusion matrix
        self._confusion_matrix.update(yi, round(yi_pred.get(True)))
        # Train the model with the new sample
        self._model.learn_one(tweet_input, yi)
        if plot_metrics:
            print(f'ROC AUC After Test Set learning: {self._metric}')
            self.plot_metrics(
                self._metric_scores,
                self._confusion_matrix
            )
        # UPDATE RECOMMENDATION -> Add this to a function
        recommended = self.recommended
        min_value_in_dict = min(recommended.values())
        if yi_pred.get(True) > min_value_in_dict:
            recommended[id] = yi_pred.get(True)
            # This should use a better and faster sorting. TODO
            keys = heapq.nlargest(
                self.n_tweets_to_show, recommended, key=recommended.get
            )
            recommended = {
                k: v for k, v in recommended.items() if k in keys
            }
            recommended = dict(
                sorted(
                    recommended.items(),
                    key=operator.itemgetter(1),
                    reverse=True
                )
            )
            min_value_in_dict = min(recommended.values())

    @property
    def _return_fake_user_input(self):
        """Simple simulation method to represent acceptance (1)
        or rejection (0) for a given tweet

        :return: a 1 or a 0, representing acceptance
        :rtype: int
        """
        return np.random.randint(low=0, high=1)

    @_return_fake_user_input.setter
    def _return_fake_user_input(self, val):
        if val in [0, 1]:
            self._fake_user_input = val
        else:
            raise ValueError('Value must be either 0 or 1')

    def check_drift(
            self
    ):
        # TODO
        raise NotImplementedError('Coming SOON')

    def save_model_pipeline(
            self,
            saving_path: str
    ):
        with open(
            saving_path,
            "wb"
        ) as f:
            pickle.dump(self._model, f)
            print(f"Model Pipeline saved at {saving_path}")
