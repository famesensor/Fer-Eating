import pandas as pd
import numpy as np


def prepare_dataframe(expression_data, behavior_data, image_data, interest_area_data):

    expression_level_dict = {
        "neutral": "neutral",
        "anger": "negative",
        "contempt": "negative",
        "disgust": "negative",
        "fear": "negative",
        "happy": "positive",
        "sadness": "negative",
        "surprise": "positive"
    }

    # convert list to dataframe
    df_expression = pd.DataFrame.from_records(expression_data)
    df_expression.columns = ['frame', 'expression']

    df_behavior = pd.DataFrame.from_records(behavior_data)
    df_behavior.columns = ['frame', 'behavior']

    df_interest_area = pd.DataFrame.from_records(interest_area_data)
    df_interest_area.columns = ['frame', 'area']

    df_image = pd.DataFrame.from_records(image_data)
    df_image.columns = ['frame', 'image']

    # combine to a dataframe
    df = pd.merge(df_expression, df_behavior, on="frame")
    df = pd.merge(df, df_interest_area, on="frame", how="left")
    df = pd.merge(df, df_image, on="frame", how="left")
    df["frame"] = df["frame"].astype(int)

    # add level of expression
    df['level'] = df['expression'].map(expression_level_dict)

    # just wrap a column for one line plot
    df['wrap'] = 'expression'

    return df


def group_expression_level(dataframe):
    # group by continuous level
    df_level = dataframe.copy()
    df_level['group'] = df_level['level'].ne(
        dataframe['level'].shift()).cumsum()
    return df_level


def group_expression(dataframe):

    expressions = ['disgust', 'fear', 'anger', 'sadness',
                   'contempt', 'neutral', 'surprise', 'happy']

    # group by continuous expression
    df_expression = dataframe.copy()
    df_expression['group'] = df_expression['expression'].ne(
        dataframe['expression'].shift()).cumsum()

    # wrap expressions for legend in graph
    index = -1

    for expression in expressions:
        df_expression.loc[index] = [np.nan, expression,
                                    np.nan, np.nan, np.nan, np.nan, 'expression', np.nan]
        index = index - 1

    df_expression.sort_index(inplace=True)

    return df_expression


def group_interest_area(dataframe):
    # group by continuous interest area
    df_interest_area = dataframe.copy()
    df_interest_area['group'] = df_interest_area['area'].ne(
        dataframe['area'].shift()).cumsum()
    df_interest_area_group = df_interest_area.groupby(
        'group')  # extract each group

    return df_interest_area_group
