from time import perf_counter
import time
from datetime import datetime
import json
import numpy as np
import pandas as pd
import io
import os
import pstats
import cProfile
import traceback
from IPython.display import display

MAX_ROWS = 50
NP_FLOAT_PRECISION = 7
NP_FLOAT_PRECISION_F = lambda x: "{0:0.7f}".format(x) # shows 7 decimal places
PD_FLOAT_PRECISION = 7 
PD_FLOAT_PRECISION_F = lambda x: "{0:0.7f}".format(x) # shows 7 decimal places
PD_MAX_COLUMNS = None # show all of the columns
PD_MAX_ROWS = 50 # show up to 50 rows instead of the default 20

######################
# PRINTING FUNCTIONS #
######################

def stack_trace(e):
    """returns a string representation of the stack trace to
    help with debugging and error messages
    """
    return "".join(traceback.TracebackException.from_exception(e).format())

def curr_time_str():
    """returns the current time as a string YYYY-MM-DD"""
    now = datetime.utcnow()
    now_str = now.strftime("%m-%d-%Y")
    return now_str

def current_time(utc=True, string=True, timestamp=False, hour_24=False):
    """prints the current time in YYYY-MM-DD HH:MM pm/am format
    can specify current utc time or current local time
    default local
    """
    if utc:
        now = datetime.utcnow()
    else:
        now = datetime.now()

    if timestamp:
        return now.timestamp()

    if not string:
        return now

    if not hour_24:
        format = "%Y-%m-%d %I:%M %p"
    else:
        format = "%Y-%m-%d %H:%M"
        
    current_time_str = now.strftime(format)
    return current_time_str

def log_print(*args, end="\n", flush=False, sep=" ", filepath="logs/live_log.txt", header=False, utc=False, only_log=False):
    """functions like a normal print, but also sends whatever is printed to
    a specified file with './print_log.txt' set as the default
    """
    string = ""
    for i, a in enumerate(args):
        string += str(a)
        if i < len(args) - 1:
            string += sep

    # print to the standard output stream
    if not only_log:
        print(string, end=end, flush=flush, sep=sep)

    # add the output to the filepath specified
    with open(filepath, "a+") as file:
        if header:
            file.write(current_time(string=True, utc=utc, timestamp=False))
            file.write("\n" + "-"*3 + "\n")
        file.write(string)
        file.write(end)
        if header:
            file.write("\n" + "-"*3 + "\n")
            
def format_perc(perc):
    """takes in a percentage as a decimal and formats it
    in the form of +x.xx% or -x.xx%
    """
    perc *= 100

    if perc >= 0:
        return f"+{perc:,.2f}%"
    else:
        return f"-{abs(perc):,.2f}%"


def format_price(price):
    """takes in a price and formats it in the form of $x,xxx,xxx.xx"""
    return f"${price:,.2f}"


def pprint(json_response, ret_string=False):
    # prints a json response in a legible format :)
    try:
        if not ret_string:
            print(json.dumps(json_response, indent=4))
            return
        else:
            return str(json.dumps(json_response, indent=4))
    except TypeError as e:
        if ret_string:
            return "TypeError: " + str(e)
        else:
            return
        

print_time_p = 0
def p(string=None):
    """a quick and efficient way to have print updates that keeps track of time
    
    Example:
    p("running this function")
    f(*args, **kwargs)
    p()
    
    will print:
    running this function...DONE 0.123 se
    """
    global print_time_p
    if string is not None:
        print(f"{string}...", end="", flush=True)
        print_time_p = perf_counter()
    else:
        print("DONE", round(perf_counter() - print_time_p, 3), "sec", flush=True)
        

###################
# PANDA FUNCTIONS #
###################

def set_np_pd_display_params(np, pd):
    """sets numpy and pandas display properties"""
    # sets numpy to avoid scientific and only round to the nth decimal
    np.set_printoptions(precision=NP_FLOAT_PRECISION, suppress=True)
    np.set_printoptions(formatter={'float': NP_FLOAT_PRECISION_F})

    # sets pandas to avoid scientific and only round to the nth decimal
    pd.set_option("display.precision", PD_FLOAT_PRECISION)
    pd.set_option('display.float_format', PD_FLOAT_PRECISION_F)

    # sets pandas to show all columns and rows
    pd.set_option('display.max_columns', PD_MAX_COLUMNS)
    pd.set_option('display.max_rows', PD_MAX_ROWS)
    

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
    
def print_display(variable_name, max_rows=None, shuffle=False):
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
    if max_rows is None and "MAX_ROWS" in globals():
        max_rows = MAX_ROWS
    
    # Get the local variables from the current scope
    local_vars = globals()

    # Check if the variable with the specified name exists
    if variable_name in local_vars:
        # Print the variable name
        print("Name:", variable_name)

        # Get the variable value using the variable name
        variable_value = local_vars[variable_name]
        
        if type(variable_value) == pd.DataFrame:
            if shuffle:
                variable_vale = variable_value.sample(frac=1)
                
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
        
def movecol(df, cols_to_move=[], ref_col='', place='After'):
    cols = df.columns.tolist()
    if place == 'After':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == 'Before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]
    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]
    return(df[seg1 + seg2 + seg3])
    
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


###########################
# FUNCTIONALITY FUNCTIONS #
###########################

def argmax_py(lst):
    return max(range(len(lst)), key=lambda x: lst[x])


def time_function(func, *args, **kwargs):
    """takes in a function, its argument, and its keyword arguments. It runs this function
    and outputs the cProfile timing analysis in the kwargs['output_path'] file. If no 'output_path'
    in the keyword arguments, then it will just save it to a file called time_output_<func_name> in
    the current directory.
    
    example: 
      time_function(your_function, arg1, arg2, kwarg1=kwarg1, kwarg2=kwarg2, output_path="path/to/file/"))
    """
    output_name = f"time_{func.__name__}"
    if 'output_path' in kwargs:
        output_path = kwargs['output_path'] + output_name
        del kwargs['output_path']
    else:
        output_path = output_name
        
    try:
        pr = cProfile.Profile()
        pr.enable()
        func(*args, **kwargs)
    except KeyboardInterrupt as ke:
        pass
    finally:
        pr.disable()
        prof_path = output_path + '.prof'
        text_path = output_path + '.txt'
        with open(prof_path, 'w+') as file:
            file.write("")
        pr.dump_stats(prof_path)

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats()
        
        print(f"Saving a record of how long {func.__name__} took to run in {output_path}.txt")

        with open(text_path, 'w+') as file:
            file.write(s.getvalue())
            
        if os.path.exists(prof_path):
            os.remove(prof_path)

            
def print_skip_exceptions(full_stack_trace=True):
    """ a decorator that will just print an exception thrown
    by the function without raising it up the call stack
    
    Ex:
    @print_skip_exceptions(True)
    def f():
        ...
        raise ValueError("foobar")
    
    f() # prints the stack trace of the exception
    """
    
    def wrap(func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            except KeyboardInterrupt as e:
                print("Caught KeyboardInterrupt and raising again")
                raise e
            except Exception as e:
                if full_stack_trace:
                    print(stack_trace(e))
                else:
                    print(e)
                return None
            
        return wrapper
    
    return wrap


def monitor_threads(*threads, path="logs/threads_running.json"):
    """ for all of the threads running from this main.py file, this will
    update the logs/threads_running.json every minute or so with a timestamp
    of the last time that each thread was running. If you ever want to know
    when a thread crashed (or if a thread has crashed) just check the
    logs/threads_running.json file
    """
    # if the logs/threads_running.json file doesn't exist, create it
    if not os.path.exists(path):
        with open(path, "w+") as file:
            file.write("{}")
            
    while True:
        for thread in threads:
            # if the current thread is still running, then udpate the
            # logs/threads_running.json file with the current time
            if thread.is_alive():
                with open(path, "r+") as file:
                    running_dict = json.load(file)
                    running_dict[thread.name] = current_time()
                    # write the updated dictionary to the file
                    file.seek(0)
                    file.truncate()
                    json.dump(running_dict, file, indent=2)
                    
            # if the thread is no longer running, then the thread entry will
            # just have the last time that the thread was running
            else:
                pass
            
        # sleep for 1 minute, so that it only checks all of the threads
        # every minute
        time.sleep(60)