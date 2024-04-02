import re
from pandas import DataFrame


def preprocess(instances: list | dict | DataFrame):
    if isinstance(instances, dict | list):
        columns = ['purpose', 'counterpart_name', 'category']
        df = DataFrame(instances, columns=columns)
    else:
        df = instances

    df['counterpart_name'] = df['counterpart_name'].fillna('')
    df['purpose'] = df['purpose'].fillna('')
    df["data"] = df["counterpart_name"] + " " + df["purpose"]
    df["data"] = df["data"].apply(remove_numbers)

    return df["data"]


def remove_numbers(text):
    return re.sub(r'\d+', '', text)
