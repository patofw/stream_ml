import pandas as pd
import numpy as np

import nltk
import textstat

from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import matplotlib as mpl
import dabl

# Set plotting styles
plt.style.use('ggplot')
mpl.rcParams['figure.figsize'] = (10, 6)


class TextAnalytics:

    def __init__(
            self,
            dataframe: pd.DataFrame,
            drop_duplicates: bool = True,
            sentiment_model: object = None,
            tokenizer: object = TweetTokenizer(),
            stopwords: set = set(stopwords.words('english')),
            plot_summary: bool = True,

    ) -> None:

        self.dataframe = dataframe.copy().reset_index(drop=True)
        self._drop_duplicates = drop_duplicates
        # Check for duplicates
        self._duplicate_check()
        self.sentiment_model = sentiment_model
        self.tokenizer = tokenizer
        self.stopwords = stopwords
        self.plot_summary = plot_summary

        # init some defaults
        self.has_sentiment: bool = False
        self.predictive_df = self.build_prediction_df

    def _tokenize(
            self,
    ):

        # Tokenize and Remove stop words
        _tokens = self.dataframe['text'].apply(
            self.tokenizer.tokenize
        )
        # # remove stopwords after tokenizing
        self.dataframe['tokens'] = _tokens.apply(
            lambda row: [
                word.lower() for word in row if word.lower()
                not in self.stopwords
            ]
        )

    def _duplicate_check(self):

        print('DUPLICATES ANALYSIS -----------------------')
        len_df = len(self.dataframe)
        df_without_dups = self.dataframe.drop(
            'embedding', axis=1
        ).drop_duplicates()

        duplicates = len_df - len(df_without_dups)

        if duplicates > 0:
            print(f'There are {duplicates} duplicate values in the DF')
            print("Most Common text duplicates and their count")
            print('----------------------------------------\n\n')
            print(self.dataframe.text.value_counts().head(10))
            if self._drop_duplicates:
                print('Dropping Duplicates\n\n')
                self.dataframe = self.dataframe.loc[
                    df_without_dups.index
                ].reset_index(drop=True)
        else:
            print('No duplicate values found\n\n')

    def _add_sentiment(self):
        """Adds a sentiment score using a sentiment analyzer.
        If none given, uses NLTK's Vader
        """
        if self.sentiment_model is None:
            # Lading default Sentiment Analysis module.
            nltk.download('vader_lexicon')
            self.sentiment_model = SentimentIntensityAnalyzer()
        # For this Exercise, A simple VADER model was selected.
        # However the pipeline should be built to use any sentiment model
        # as an input

        self.dataframe['sentiment_score'] = self.dataframe['text'].apply(
            lambda x: self.sentiment_model.polarity_scores(x)['compound']
        )

        # TextBlob sentiment
        self.dataframe['textblob_sentiment'] = self.dataframe['text'].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        # when sentiment added, we don't re run it
        self.has_sentiment = True

    def _reading_ease_plots(
            self,
    ):
        """Plots the reading ease EDA analysis
        """
        # Reading Time and Label
        sns.histplot(
                data=self.predictive_df.reset_index(drop=True),
                x='reading_time',
                hue='label',
                bins=20,
                kde=True,
            )
        plt.xlabel('Reading Time Norm.')
        plt.ylabel('Frequency')
        plt.title('Reading Time Score Distribution')
        plt.show()
        # Reading ease plot and Label
        sns.histplot(
                data=self.predictive_df.reset_index(drop=True),
                x='reading_ease',
                hue='label',
                bins=20,
                kde=True,
            )
        plt.xlabel('Reading Ease Score Norm.')
        plt.ylabel('Frequency')
        plt.title('Reading Ease Score Distribution')
        plt.show()

    def _sentiment_summary(self):
        if not self.has_sentiment:
            self._add_sentiment()

        if self.plot_summary:

            # Distribution of sentiment scores per label
            plt.figure(figsize=(10, 6))
            sns.histplot(
                data=self.dataframe,
                x='textblob_sentiment',
                hue='label',
                bins=20,
                kde=True,
            )
            plt.xlabel('TextBlob Sentiment Score')
            plt.ylabel('Frequency')
            plt.title('Sentiment Score Distribution')
            plt.show()

            # Distribution of sentiment scores per topic
            plt.figure(figsize=(10, 6))
            sns.histplot(
                data=self.dataframe,
                x='textblob_sentiment',
                hue='topic',
                bins=20,
                kde=True,
            )
            plt.xlabel('TextBlob Sentiment Score')
            plt.ylabel('Frequency')
            plt.title('Sentiment Score Distribution')
            plt.show()

    def basic_summary(
            self,
    ):
        """Creates a basic summary with sentiment, easyness of read
        and predictive power metrics.
        """
        if 'tokens' not in self.dataframe.columns:
            # make sure it is tokenized before launching
            # the sentiment analysis.
            self._tokenize()

        print('\n\nAcceptance Rate per topic')
        print(self._acceptance_rates())
        print('....................................................')
        print('----------------------------------------------------\n\n')
        if self.plot_summary:
            # Plotting top tokens
            accepted = self.dataframe.query('label == "accepted"')['tokens']

            accepted = [
                word for tokens in accepted
                for word in tokens if len(word) > 1
            ]

            # Plot
            nltk.FreqDist(accepted).plot(
                20, color='darkgreen',
                title='Top 20 Most Common Words in Accepted Tweets'
            )  # most_common(10)
            plt.show()

            # Not accepted
            rejected = self.dataframe.query('label != "accepted"')['tokens']

            rejected = [
                word for tokens in rejected
                for word in tokens if len(word) > 1
            ]
            # Plot
            nltk.FreqDist(rejected).plot(
                20, color='darkred',
                title='Top 20 Most Common Words in Rejected Tweets'
            )  # most_common(10)

            plt.show()
            # Sentiment related plots
            self._sentiment_summary()

            print('Reading Ease and Time Analysis')
            self._reading_ease_plots()

            print('-----------------------------------------------------')
            print('Linear Discriminant Analysis')

            self.linear_discriminant_analysis()

    @property
    def build_prediction_df(self):
        """Builds a dataframe for prediction of acceptance rate

        :return: DataFrame ready to be fed into the OnlineTraining Class
        :rtype: pd.Dataframe
        """
        pred_df = pd.DataFrame(
            np.array(
                [emb for emb in self.dataframe.embedding.values]
            )
        )
        # Adding the topics as a dummy variable
        topic_dummies = pd.get_dummies(self.dataframe.topic)
        pred_df = pd.concat([pred_df, topic_dummies], axis=1)

        # Adding reading ease and time to read figures.
        pred_df['reading_ease'] = self.dataframe['text'].apply(
            # reading ease score / max value.
            lambda x: textstat.flesch_reading_ease(x) / 100
        )
        pred_df['reading_time'] = self.dataframe['text'].apply(
            # reading time / 8 (max time for short texts)
            lambda x: textstat.reading_time(x) / 8
        )
        # adding the target variable
        pred_df['label'] = self.dataframe.label

        pred_df.columns = [str(col) for col in pred_df.columns]

        return pred_df

    def linear_discriminant_analysis(self):
        """Uses Dabl to plot predictive potential using a
        linear discriminant analysis
        """

        dabl.plot(
            self.predictive_df,
            target_col='label'
        )
        plt.show()

    def _acceptance_rates(self):
        return self.dataframe.groupby(
            'topic'
        ).apply(
            lambda g: round(g.label.value_counts() / len(g), 2))


if __name__ == '__main__':
    from utils import read_data_input

    df = read_data_input("./data/queries")  # load data
    # Launch The text analytics Pipeline
    txt_analytics = TextAnalytics(df)
    txt_analytics.basic_summary()
