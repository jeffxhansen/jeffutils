from collections import defaultdict
import os
import io

import sqlite3 as sql
import numpy as np
import pandas as pd
from IPython.display import display

import pstats
import cProfile

MAX_ROWS = 50
NP_FLOAT_PRECISION = 7
NP_FLOAT_PRECISION_F = lambda x: "{0:0.7f}".format(x) # shows 7 decimal places
PD_FLOAT_PRECISION = 7 
PD_FLOAT_PRECISION_F = lambda x: "{0:0.7f}".format(x) # shows 7 decimal places
PD_MAX_COLUMNS = None # show all of the columns
PD_MAX_ROWS = 50 # show up to 50 rows instead of the default 20


MAX_ROWS = 50
NP_FLOAT_PRECISION = 7
NP_FLOAT_PRECISION_F = lambda x: "{0:0.7f}".format(x) # shows 7 decimal places
PD_FLOAT_PRECISION = 7 
PD_FLOAT_PRECISION_F = lambda x: "{0:0.7f}".format(x) # shows 7 decimal places
PD_MAX_COLUMNS = None # show all of the columns
PD_MAX_ROWS = 50 # show up to 50 rows instead of the default 20


############################################################
#                         PROFILING                        #
############################################################

def time_function(func, args=None, kwargs=None, output_directory=None):
    """takes in a function, its argument, and its keyword arguments. It runs this function
    and outputs the cProfile timing analysis in the 'output_directory' file. If no 'output_directory'
    is provided, then it will just save it to a file called time_<func_name> in
    the current directory.
    
    example: 
      time_function(your_function, args=(arg1, arg2,), kwargs={'kwarg1':kwarg1, 'kwarg2':kwarg2}, output_directory="path/to/file/"))
    """
    # setup where the output will be saved
    output_name = f"time_{func.__name__}"
    if output_directory is not None:
        output_path_dir = os.path.dirname(output_directory)
        output_path = os.path.join(output_path_dir, output_name)
    else:
        output_path = output_name
    # make sure the directory for the output_path exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # run the function and save the cProfile timing analysis
    try:
        pr = cProfile.Profile()
        pr.enable()
        if args is not None and kwargs is not None:
            func(*args, **kwargs)
        elif args is not None:
            func(*args)
        elif kwargs is not None:
            func(**kwargs)
        else:
            func()
    
    # make sure that it still outputs the cProfile timing analysis even if the function
    # is interrupted by a KeyboardInterrupt
    except KeyboardInterrupt as ke:
        pass
    finally:
        # the cProfile has to go to a .prof file
        pr.disable()
        prof_path = output_path + '.prof'
        text_path = output_path + '.txt'
        with open(prof_path, 'w+') as file:
            file.write("")
        pr.dump_stats(prof_path)

        # load the .prof file into a text file
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats()
        
        print(f"Saving a record of how long {func.__name__} took to run in {output_path}.txt")

        with open(text_path, 'w+') as file:
            file.write(s.getvalue())
            
        # remove the .prof path since it is no longer needed
        if os.path.exists(prof_path):
            os.remove(prof_path)

###################
# PANDA FUNCTIONS #
###################

def set_np_pd_display_params(np, pd, params=None):
    """sets numpy and pandas display properties"""
    if params is None:
        params = {}
    # sets numpy to avoid scientific and only round to the nth decimal
    np.set_printoptions(precision=params.get('NP_FLOAT_PRECISION', NP_FLOAT_PRECISION), suppress=True)
    np.set_printoptions(formatter={'float': params.get('NP_FLOAT_PRECISION_F', NP_FLOAT_PRECISION_F)})

    # sets pandas to avoid scientific and only round to the nth decimal
    pd.set_option("display.precision", params.get('PD_FLOAT_PRECISION', PD_FLOAT_PRECISION))
    pd.set_option('display.float_format', params.get('PD_FLOAT_PRECISION_F', PD_FLOAT_PRECISION_F))

    # sets pandas to show all columns and rows
    pd.set_option('display.max_columns', params.get('PD_MAX_COLUMNS', PD_MAX_COLUMNS))
    pd.set_option('display.max_rows', params.get('PD_MAX_ROWS', PD_MAX_ROWS))
    

def generate_df(rows, columns, low=0, high=50, np_arr=False):

    arr = np.random.randint(low, high, size=(rows, columns))
    if np_arr:
        return arr
    df = pd.DataFrame(arr, columns=[chr(i)
                      for i in range(ord("A"), ord("A")+columns)])
    return df


def fill_nan_with_empty_list(df, column, inplace=True):
    
    if not inplace:
        df = df.copy()
        
    nan_rows = df[column].isna()
    n = np.sum(nan_rows)
    empty_lists = pd.Series(np.empty((n,0)).tolist(), index=df.index[nan_rows])
    df.loc[nan_rows, column] = empty_lists
    
    if not inplace:
        return df[column]
    
def print_display(variable_name, variables, max_rows=None, shuffle=False):
    """ this is a better display function where you can specify how many rows to
    show for the dataframe, and it also prints the dataframe name
    
    parameters:
    variable_name: name of the variable to display (must be a global variable)
    max_rows: maximum number of rows to display
        if the dataframe is larger than the max_rows, it will display the first
        half, then a row of '...', then the last half
    shuffle: whether to shuffle the dataframe before displaying
        this allows you to see random rows in the middle instead of just the first
        few rows and then the last few rows
    """
    if max_rows is None and "MAX_ROWS" in variables:
        max_rows = MAX_ROWS
    
    # Get the local variables from the current scope
    local_vars = variables

    # Check if the variable with the specified name exists
    if variable_name in local_vars:
        # Print the variable name
        print("Name:", variable_name)

        # Get the variable value using the variable name
        variable_value = local_vars[variable_name]
        
        if type(variable_value) == pd.DataFrame:
            if shuffle:
                variable_value = variable_value.sample(frac=1)
                
            if max_rows is not None and len(variable_value) > max_rows:
                head_tail = pd.concat([
                    variable_value.head(max_rows//2),
                    pd.DataFrame({col:'...' for col in variable_value.columns}, index=['...']),
                    variable_value.tail(max_rows//2)
                ])
                display(head_tail)
            else:
                display(variable_value)
            
            df_shape = variable_value.shape
            print(f"{df_shape[0]}x{df_shape[1]}")
        else:
            print(variable_value)
        
    else:
        print(f"Variable '{variable_name}' not found.")
        
def movecol(df, cols_to_move=None, ref_col='', place='After', make_copy=True) -> pd.DataFrame:
    """
    Move columns within a DataFrame to a specified position relative to a reference column.

    Parameters:
    - df (DataFrame): The DataFrame containing the columns to be moved.
    - cols_to_move (list): A list of column names to be moved.
    - ref_col (str): The name of the reference column relative to which the specified columns will be moved.
    - place (str): The position where the specified columns will be placed relative to the reference column. 
      It can be either 'After' or 'Before'.\
    - make_copy (bool): Whether to return a new DataFrame or modify the original DataFrame in place.

    Returns:
    - DataFrame: The DataFrame with the specified columns moved to the desired position relative to the reference column.

    Example:
    If df is a DataFrame with columns 'A', 'B', 'C', 'D' and you want to move columns 'B' and 'C' 
    after column 'A', you can use:
    
    movecol(df, cols_to_move=['B', 'C'], ref_col='A', place='After')
    """
    if make_copy:
        df = df.copy()
    
    # If cols_to_move is not provided, set it as an empty list
    if cols_to_move is None:
        cols_to_move = []
        
    # Get the list of columns in the DataFrame
    cols = df.columns.tolist()
    
    # move the columns around and return the dataframe
    if place == 'After':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == 'Before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]
    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]
    return df[seg1 + seg2 + seg3]

def rename_duplicate_columns(df):
    """ renames duplicate columns in a pandas dataframe by appending a number to the end """
    df = df.copy()
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values[1:]] = [f"{dup}{i}" for i in range(2, sum(cols == dup)+2-1)]
    df.columns = cols
    return df

    
###################
# NUMPY FUNCTIONS #
###################

def rolling_window(a, window=15):
    """returns a numpy rolling window
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return rolling


def np_moving_average(a, window=3, include_nan=True):
    """uses numpy to compute the weighted average"""
    n = window
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    if include_nan:
        fill = np.full(n-1, np.nan)
        return np.concatenate([fill, ret[n - 1:] / n])
    else:
        return ret[n - 1:] / n
    

def np_map(arr, map):
    """ maps all values in a numpy array based on a python dictionary """
    return np.vectorize(map.__getitem__)(arr)

        
############################################################
#                       SQL FUNCTIONS                      #
############################################################

def get_sql_tables_info(db_path, verbose=True):
    """
    Retrieve and print all table names and their respective column names from a SQLite database.

    Args:
        db_path (str): The file path to the SQLite database.
        verbose (bool): If True, prints the table names and columns to the console. Default is True.

    Returns:
        dict: A dictionary where the keys are table names and the values are lists of column names for each table.
    
    Example:
        >>> sql_table_names_and_cols('example.db', verbose=True)
        table1: id, name, age
        table2: id, department, salary
        {
            'table1': {
                'id': {'cid': 0, 'name': 'id', 'type': 'INTEGER', 'notnull': 0, 'dflt_value': None, 'pk': 0},
                'name': {'cid': 1, 'name': 'name', 'type': 'TEXT', 'notnull': 0, 'dflt_value': None, 'pk': 0},
                'age': {'cid': 2, 'name': 'age', 'type': 'INTEGER', 'notnull': 0, 'dflt_value': None, 'pk': 0}
            }
            'table2': {
                'id': {'cid': 0, 'name': 'id', 'type': 'INTEGER', 'notnull'},
                'departmet': {'cid': 1, 'name': 'department', 'type': 'TEXT', 'notnull': 0, 'dflt_value': None, 'pk': 0},
                'salary': {'cid': 2, 'name': 'salary', 'type': 'INTEGER', 'notnull': 0, 'dflt_value': None, 'pk': 0}
            }
        }
    
    Note:
        Ensure the database file exists at the specified path before calling this function.
    """
    # make sure the db_path exists and is a sqlite .db file
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at path: {db_path}")
    if not db_path.endswith('.db'):
        raise ValueError("Database file must be a SQLite file with a .db extension.")
    
    try:
        table_info = defaultdict(dict)
        
        with sql.connect(db_path) as conn:
            cur = conn.cursor()
            
            # extrac each of the table names
            query = '''
            SELECT name FROM sqlite_master WHERE type='table'
            '''
            result = cur.execute(query).fetchall()
            
            # fore each table extract the table column information
            for row in result:
                table_name = row[0]
                
                # use PRAGMA table_info to get the column names
                # and all of the information for each column
                query = (
                    f"PRAGMA table_info({table_name}) "
                )
                result = cur.execute(query).fetchall()
                for row in result:
                    cid, name, type, notnull, dflt_value, pk = row
                    entry = {
                        'cid': cid,
                        'name': name,
                        'type': type,
                        'notnull': notnull,
                        'dflt_value': dflt_value,
                        'pk': pk
                    }
                    table_info[table_name][name] = entry
            
    finally:
        conn.close()
        
    if verbose:
        for table_name, col_names in table_info.items():
            print(f'{table_name}:', ", ".join(map(str, col_names)))
            
    return dict(table_info)

def get_unique_counts(path_db, table_name, ref_col, cols):
    """
    Retrieve the count of unique values for specified columns grouped by a reference column from a SQLite table.

    Args:
        path_db (str): Path to the SQLite database file.
        table_name (str): Name of the table in the database.
        ref_col (str): Name of the reference column to group by.
        cols (list): List of column names for which to count unique values.

    Returns:
        pandas.DataFrame: DataFrame containing the count of unique values for each specified column grouped by the reference column.

    Example:
        Given input data in GameTest table:
        game_id  A  B
        0         0  a  a
        1         0  a  a
        2         0  a  b
        3         0  a  b
        4         0  a  b
        5         1  b  c
        6         1  b  c
        7         1  b  c
        8         1  b  c
        9         1  b  d
        10        1  b  d
        11        1  b  e

        Calling get_unique_counts('your_database.db', 'GameTest', 'game_id', ['A', 'B']) returns:
           game_id  unique_A  unique_B
        0        0         1         2
        1        1         1         3
    """
    if not os.path.exists(path_db):
        raise FileNotFoundError(f"Database file not found: {path_db}")
    if not path_db.endswith('.db'):
        raise ValueError("Database file must have a .db extension")
    
    try:
        with sql.connect(path_db) as conn:
            cur = conn.cursor()
            query = (
                f"SELECT {ref_col}, " + 
                    ', '.join([f'COUNT(DISTINCT {col}) AS unique_{col}' for col in cols]) + 
                f" FROM {table_name} \n"
                f"GROUP BY {ref_col}"
            )
            res_df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
        
    return res_df

def get_sql_table_as_df(db_path, table_name):
    """ Get the table as a pandas DataFrame
    
    Args:
    db_path (str): The path to the database file
    table_name (str): The name of the table
    
    Returns:
    pd.DataFrame: The table as a DataFrame
    """
    # validate the input
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file {db_path} does not exist")
    if not db_path.endswith(".db"):
        raise ValueError(f"Table name {table_name} should end with .db")
    
    # get the table as a DataFrame
    try:
        with sql.connect(db_path) as conn:
            return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()
