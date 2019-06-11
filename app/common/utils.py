import glob
import pandas as pd


def read_csv_from_folder(path_pattern: str, file_limit: int = None) -> pd.DataFrame:
    """Read csv files from path pattern. Only read file_limit files if it is defined"""
    file_names = glob.glob(path_pattern)
    file_content_list = []

    limit = len(file_names)
    if file_limit is not None:
        limit = min(file_limit, limit)

    print('Reading {} files from "{}" ...'.format(limit, path_pattern))

    for file_name in file_names[:limit]:
        df = pd.read_csv(file_name, index_col=None, header=0)
        file_content_list.append(df)

    contents = pd.concat(file_content_list, axis=0, ignore_index=True)
    del file_content_list
    return contents
