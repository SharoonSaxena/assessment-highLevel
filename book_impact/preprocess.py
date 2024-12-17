import tomllib
import numpy
import pandas as pd
from pandas import DataFrame, Series
from joblib import Parallel, delayed
from tqdm import tqdm


def clean_authors(author_list: str) -> str:
    """
    Cleans author list string by removing quotes and brackets.

    Args:
        author_list (str): Raw author list string containing quotes and brackets

    Returns:
        str: Cleaned author list string
    """
    return author_list.replace("'", "").replace("[", "").replace("]", "")


def extract_year(date_str: str) -> str:
    """
    Extracts year from date string.

    Args:
        date_str (str): Date string in format 'YYYY-MM-DD'

    Returns:
        str: Extracted year
    """
    return date_str.split("-")[0]


def date_cleaner(string: str) -> str:
    """
    Extracts and validates 4-digit year from string.

    Args:
        string (str): Input string containing date information

    Returns:
        str or numpy.nan: 4-digit year if valid, numpy.nan otherwise
    """
    numbers = "".join(filter(str.isdigit, string))
    if len(numbers) == 4:
        return numbers
    else:
        return numpy.nan


def category_cleaner(category_string: str) -> str:
    """
    Cleans category string by removing quotes, brackets and converting to lowercase.

    Args:
        category_string (str): Raw category string

    Returns:
        str: Cleaned lowercase category string
    """
    return category_string.replace("[", "").replace("]", "").replace("'", "").lower()


def stringify_single(record: dict) -> str:
    """
    Converts a dictionary record to a formatted string.

    Args:
        record (dict): Dictionary containing record fields

    Returns:
        str: Formatted string with key-value pairs
    """
    string = ""
    for key, value in record.items():
        string += f"{key} - {value}; "
    return string


def stringify_df(df: DataFrame) -> list:
    """
    Converts DataFrame records to list of formatted strings.

    Args:
        df (DataFrame): Input DataFrame

    Returns:
        list: List of formatted strings for each record
    """
    records = df.to_dict("records")
    return [stringify_single(record) for record in records]


def preprocess_pipe(df: DataFrame, config: dict) -> tuple[list, list]:
    """
    Main preprocessing pipeline for DataFrame.

    Args:
        df (DataFrame): Input DataFrame
        config (dict): Configuration dictionary with preprocessing parameters

    Returns:
        tuple[list, list]: Preprocessed features (X) and labels (y)
    """
    # Basic preprocessing steps
    df = df.drop(config["drop_columns"], axis=1)
    df["description"] = df["description"].fillna(config["missing_string"])
    df["authors"] = df["authors"].fillna(config["missing_string"])
    df["publishedDate"] = df["publishedDate"].fillna(config["missing_string"])
    df["authors"] = df["authors"].map(clean_authors)
    df["publishedDate"] = df["publishedDate"].map(extract_year).map(date_cleaner).dropna()
    df["categories"] = df["categories"].map(category_cleaner)

    # Model specific preprocessing
    x_df = df.drop(columns=["Impact"], axis=1)
    y = df["Impact"].copy().to_list()
    x = x_df.to_dict("records")
    string_x = stringify_df(x_df)
    return string_x, y


def parallel_preprocess(df: DataFrame, config: dict, workers: int = 2, chunks: int = 2) -> tuple[list, list]:
    """
    Parallel preprocessing of DataFrame using multiple workers.

    Args:
        df (DataFrame): Input DataFrame
        workers (int): Number of parallel workers
        chunks (int): Number of chunks to split DataFrame
        config (dict): Preprocessing configuration

    Returns:
        tuple[list, list]: Combined preprocessed features (X) and labels (y)
    """
    chunked_df = numpy.array_split(df, chunks)

    taskq = tqdm(
        [delayed(preprocess_pipe)(chunk, config) for chunk in chunked_df],
        total=len(chunked_df),
        desc=f"parallel processing: chunks-{chunks}; processes-{workers}",
    )

    with Parallel(n_jobs=workers, verbose=0) as parallel:
        chunk_xy = parallel(taskq)

    x, y = [], []
    for chunk in chunk_xy:
        chunk_x, chunk_y = chunk
        x.extend(chunk_x)
        y.extend(chunk_y)
    return x, y
