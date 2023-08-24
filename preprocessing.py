import pandas as pd
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from constants import *


def translate_attr_dict(df, **kwargs):
    """
    For every keyword, translate the categorical value in the corresponding colymn of df to a numerical value
    according to the dictionary supplied with the keyword.

    :param df: Dataframe whose columns need to be translated
    :param kwargs: Dictionary of keyword: dictionary pairs. Keyword: Column to be translated, Dictionary: How to
    translate categories.

    :return: Dataframe with translated columns
    """
    for kw in kwargs:
        df[kw] = df[kw].apply(lambda x: kwargs[kw][x])

    return df


def column_preprocess(df, processor, *args, transform=lambda col: col):
    """
    Preprocess columns using any model (processor).

    :param df: Dataframe whose columns are to be processed.
    :param processor: Model being used (Scaler, Encoder, etc.)
    :param args: All columns in df to be processed
    :param transform: Transformation to columns if needed.
    :return: df with altered columns
    """
    for arg in args:
        df[arg] = processor.fit_transform(transform(df[arg]))

    return df


def convert_numeric(df: pd.DataFrame, **kwargs):
    """
    Convert all of df columns to numeric values

    :param df: Dataframe whose values are being converted
    :param kwargs: All columns that need to be translated from categorical values into numerical values + translations.
    :return: Dataframe whose columns are all numeric.
    """
    translate_attr_dict(df, **kwargs)
    column_preprocess(df, LabelEncoder(), *df.columns)
    # Convert all values into numbers between 0 and 1:
    column_preprocess(df, MaxAbsScaler(), *df.columns, transform=lambda col: col.to_numpy().reshape(-1, 1))
    return df


def create_sample(df: pd.DataFrame, target, sample_size=SAMPLE, train=TRAIN, val=VALIDATION):
    """
    Create train and test samples, such that there is no label imbalance of target (if df is big enough).

    :param df: Dataframe that will be split
    :param target: Column that will serve as target column
    :param sample_size: # of rows of each type to take.
    :param train: Percentage of the new dataframe that will be taken as train set
    :param val: Percentage of the new dataframe that will be taken as validation set
    :return: Six dataframes: X_train, X_val, X_test, y_train, y_val, y_test
    """
    df_true, df_false = df[df[target] == 1], df[df[target] == 0]  # Split df into different sets where value is 1 / 0
    if len(df_true) < sample_size or len(df_false) < sample_size:
        y = df.pop(target)  # Remove target column
        split = train_test_split(df, y, train_size=train+val)  # Split df into train and test. Not equal sizes.
    else:
        new_df = pd.concat((df_true.iloc[:sample_size], df_false.iloc[:sample_size]))  # Create sample of equal numbers
        new_df = new_df.sample(frac=1)  # Shuffle rows of dataframe so that it's not a list of 1s then 0s
        y = new_df.pop(target)  # Remove target column
        split = train_test_split(new_df, y, train_size=train+val)  # Split new df into train and test, equal amounts.
    second_split = train_test_split(split[0], split[2], train_size=train / (train + val))  # Train and validation
    return second_split[0], second_split[1], split[1], second_split[2], second_split[3], split[3]


def create_tensors(*args, dtype=torch.float32):
    """
    For every dataframe in args, create pytorch tensor.

    :param args: List of tuples of dataframes and number. If number=0 do nothing. Else convert to one hot vector with (number) classes.
    :param dtype: Types of values in each dataframe.
    :return: List of pytorch tensors
    """
    return [torch.tensor(arg.values, dtype=dtype) for arg in args]


def create_data_loaders(*args, batch=BATCH_SIZE):
    """
    For every triplet of two dataframes and boolean in args, create pytorch Data Loader

    :param args: List of tuples of (dataframe, dataframe, boolean). First dataframe: feature matrix,
    Second dataframe: target values, Boolean: shuffle (whether or not to shuffle values between batches)
    :param batch: Batch size of data loader
    :return: List of pytorch dataloaders
    """
    return [DataLoader(TensorDataset(X, y), batch, shuffle) for X, y, shuffle in args]


def preprocess_data():
    """
    This is the only function that is reliant on the specific task. Reads the CVD file, splits into train,
    validation and test and returns data loaders.
    The task: Return probability that someone will have diabetes based on their features

    :return: Data loaders for train, validation, test
    """
    df = pd.read_csv(FILE)
    df = convert_numeric(df, Age_Category=translations_age, Diabetes=translations_diabetes,
                         Checkup=translations_checkup, General_Health=translations_general_health)
    samples = create_sample(df, 'Diabetes')
    tensors = create_tensors(*samples)
    return create_data_loaders((tensors[0], tensors[3], True),
                               (tensors[1], tensors[4], False),
                               (tensors[2], tensors[5], False))
