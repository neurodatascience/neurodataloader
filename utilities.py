import pandas as pd
import os
from collections import defaultdict
from collections import defaultdict as dd
from pathlib import Path
from string import Formatter

def csv_split(table_file: str, output_template: str, column_key_translate : defaultdict = None, chunksize: int = 1000,
              include_match: defaultdict = None):
    '''
    Divides a large tabular file into single-entry tables.
    Parameters
    ----------
    table_file: str
        Name of the file containing the data to split
    output_template: str
        Format string for output files. Entries must have values defined in the table. Leading directories will be
        created if they don't already exist.
        For column names with invalid format specifiers (e.g. '.'), use 'column_key_translate' to define a mapping.
        E.g. 'example/other/{COLUMN}_table.tsv'.
    column_key_translate : defaultdict
        Optional. Translation between the field in the format string and the column name. This is mostly useful in the
        case where a field (column name) has characters that are disallowed in Python's format strings (e.g.: .)
        E.g. if your CSV has a field '3-0.0' and you want to use its value in the name, you can define a mapping:
        output_template = sub-{eid}_300-{3-0}
        column_key_translate = {'3-0','3-0.0'}
    chunksize : int
        Optional. Number of rows of the table to load at a time.
    include_match : defaultdict
        Optional. If defined, only extract rows that match all values.
    Returns
    -------
    None
    '''

    # Define behaviour
    if(column_key_translate is None):
        column_key_translate = dd()

    # Get data
    csv = pd.read_csv(table_file, chunksize=chunksize, low_memory=False)
    # Get formatting keys
    output_keys = set([parsed[1] for parsed in Formatter().parse(output_template) if parsed[1] is not None])

    # Get leading path; will need it later to make directories
    if(os.sep in output_template):
        parent_path = output_template[:output_template.rfind(os.sep)]
    else:
        parent_path = ''

    # Iterate through csv chunks
    for chunk in csv:
        # Iterate through rows in chunk
        for ind in chunk.index:
            output_keyvals = {}
            # loc[ind] returns a Series, and we need ind:ind since pandas doesn't do Pythonic indexing
            row = chunk.loc[ind:ind]
            col_translate_keys = column_key_translate.keys()

            # Build formatting dictionary for our output filenames
            for k in output_keys:
                if(k in col_translate_keys):
                    col_key = column_key_translate[k]
                else:
                    col_key = k
                output_keyvals[k] = row[col_key].iloc[0]
            is_match = True
            for k, v in include_match.items():
                if row[k].iloc[0] == v:
                    continue
                else:
                    is_match = False
                    break
            if(not is_match):
                continue
            output_name = output_template.format(**output_keyvals)
            path_name = parent_path.format(**output_keyvals)
            # Make path if it doesn't exist
            if(not Path(path_name).is_dir()):
                Path(path_name).mkdir(parents=True)
            # Output row
            row.to_csv(output_name, header=True, index=False)
    return

