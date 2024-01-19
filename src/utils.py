import os
import json

import pandas as pd

# Dir should usually come from a yml file
# or similar, but for this exercise we keep it here
DIR = "../data/hiring_queries/"


def read_data_input(
        path: str
):
    """Loads the input data from a given
    directory

    :param path: Directory of the files
    :type path: str
    :return: Dataframe with all the queries
    :rtype: pd.DataFrame
    """

    # Initialize an empty list to store the data
    df = pd.DataFrame()
    # Iterate through each file in the folder
    for filename in os.listdir(path):
        if filename.endswith('.json'):
            file_path = os.path.join(path, filename)
            with open(file_path, 'r') as file:
                file_data = json.load(file)
                _tmp = pd.DataFrame(file_data)
                # adding the topic name
                _tmp['topic'] = filename.replace('.json', '')
                df = pd.concat([df, _tmp])
    return df
