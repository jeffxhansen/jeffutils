from time import perf_counter
import time
import threading
import json
import os
import sys
import importlib
import re
import traceback
from copy import deepcopy
import ast
from datetime import datetime, timezone

import pkgutil

# this specifies hwo long the live_log.txt lines should be when calling
# the log_func_vars function
LENGTH_LIM = 100

######################
# PRINTING FUNCTIONS #
######################

def stack_trace(e):
    """returns a string representation of the stack trace to
    help with debugging and error messages
    """
    return "".join(traceback.TracebackException.from_exception(e).format())

def curr_time_str(format="%Y-%m-%d %H:%M:%S"):
    """returns the current time as a string MM-DD-YYYY"""
    now = datetime.now(timezone.utc)
    now_str = now.strftime(format)
    return now_str

def current_time(utc=True, string=True, timestamp=False, hour_24=False, seconds=False):
    """prints the current time in YYYY-MM-DD HH:MM pm/am format
    can specify current utc time or current local time
    default local
    """
    if utc:
        now = datetime.now(timezone.utc)
    else:
        now = datetime.now()

    if timestamp:
        return now.timestamp()

    if not string:
        return now

    if seconds:
        if not hour_24:
            format = "%Y-%m-%d %I:%M:%S %p"
        else:
            format = "%Y-%m-%d %H:%M:%S"
    else:
        if not hour_24:
            format = "%Y-%m-%d %I:%M %p"
        else:
            format = "%Y-%m-%d %H:%M"
        
    current_time_str = now.strftime(format)
    return current_time_str

def log_print(*args, end="\n", flush=False, sep=" ", filepath="logs/live_log.txt", 
              header=True, utc=False, only_log=False, seconds=False):
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
        
    # Get the directory of the filepath
    directory = os.path.dirname(filepath)

    # Create the directory if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # add the output to the filepath specified
    with open(filepath, "a+") as file:
        if header:
            file.write(current_time(string=True, utc=utc, timestamp=False, seconds=seconds))
            file.write("\n" + "-"*3 + "\n")
        file.write(string)
        file.write(end)
        if header:
            if end == "\n":
                file.write("-"*3 + "\n")
            else:
                file.write("\n" + "-"*3 + "\n")
            

def get_log_dict(var_names, globals, locals):
    """ takes in a list of strings that represent variable names, and 
    takes in the locals() dictionary. Returns a dictionary of the form:
    {
        var_name1: var_value1,
        var_name2: var_value2,
    }
    """
    var_dict = {**globals, **locals}
    return {
        var_name: var_dict.get(var_name, "NA") 
        for var_name in var_names
    }

def get_log_string(func_name, info_dict):
    """ returns a log_string of the form:
    
    func_name: key1: value1, key2: value2, 
        key3: value3, key4: value4,
    """
    string = ""
    curr_line = f"{func_name}: "
    
    for key, value in info_dict.items():
        curr = f"{key}: {value}, "
        # if adding this info value would make the line too long, wrap it to the next line
        if len(curr_line + curr) > LENGTH_LIM:
            string += curr_line + "\n    "
            curr_line = ""
        curr_line += curr
    
    return string + curr_line

def log_func_vars(func_name, vars, globals, locals, header=True, only_log=False):
    """
    Logs the information in a formatted way.

    Parameters:
    func_name (str): The title of the function that is making this log.
    vars (list): a list of strings that represent the variable names to log.
    globals (dict): The global variables dictionary, typically passed as globals().
    locals (dict): The local variables dictionary, typically passed as locals().
    header (bool, optional): Whether to print a header before the log string. Defaults to True.
    
    the logged string looks like:
    func_name: var_name1: var_val1, var_name2: var_val2, 
        var_name3: var_val3, var_name4: var_val4,
    
    Returns:
    str: The formatted log string.

    Notes:
    - The function will default to logging the information unless the LOG_TOGGLE dictionary has the function name set to False.
    """
    info_dict = get_log_dict(vars, globals, locals)
    string = get_log_string(func_name, info_dict)
    log_print(string, header=header, only_log=only_log)
    
    return string

def initialize_log_func_vars_func(LOG_TOGGLE):
    """ returns a function that will log the variables of a function
    with the LOG_TOGGLE dictionary set to LOG_TOGGLE
    """
    def log_func_vars_func(func_name, vars, globals, locals, header=True):
        return log_func_vars(func_name, vars, globals, locals, header=header, only_log=LOG_TOGGLE)
    
    return log_func_vars_func
            
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
        
        
def format_text_to_lines(input_text, max_line_length=60):
    """
    Format the input text into lines with a maximum line length of 'max_line_length'.

    Args:
        input_text (str): The input text to be formatted.
        max_line_length (int): Maximum desired length of each line.

    Returns:
        str: The formatted text with newlines inserted appropriately.
    """
    words = input_text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 > max_line_length:  # Check if adding this word exceeds the max line length
            lines.append(current_line)  # Add the current line to the list of lines
            current_line = word  # Start a new line with the current word
        else:
            if current_line:  # Add a space if the current line is not empty
                current_line += " "
            current_line += word  # Add the word to the current line

    # Add the last line to the list of lines
    if current_line:
        lines.append(current_line)

    # Join the lines with newline characters to form the formatted text
    formatted_text = "\n".join(lines)

    return formatted_text

###########################
# FUNCTIONALITY FUNCTIONS #
###########################

def str_to_list_py(string):
    try:
        # Using ast.literal_eval to safely evaluate string as Python literal
        parsed_list = ast.literal_eval(string)
        if isinstance(parsed_list, list):
            return parsed_list
        else:
            raise ValueError("Input is not a valid string representation of a list.")
    except (SyntaxError, ValueError) as e:
        print(f"Error: {e}")
        return None

def argmax_py(lst):
    return max(range(len(lst)), key=lambda x: lst[x])


def dict_update(d:dict, d_:dict, inplace=False):
    """ takes in two dictionaries and adds the key-value pairs from the second 
    dictionary to the first dictionary. If inplace is False, then it will return
    a new dictionary with the updated key-value pairs. If inplace is True, then
    it will update the first dictionary with the key-value pairs from the second
    and return None
    """
    if not inplace:
        d2 = deepcopy(d)
        d2.update(d_)
        return d2
    else:
        d.update(d_)

            
def print_skip_exceptions(full_stack_trace=True, log_error=True, default_return=None):
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
                    string = stack_trace(e)
                else:
                    string = e
                if log_error:
                    log_print(string)
                else:
                    print(string)
                return default_return
            
        return wrapper
    
    return wrap


def monitor_threads(threads, path="logs/threads_running.json", sleep_duration=60):
    """ for all of the threads running from this main.py file, this will
    update the logs/threads_running.json every minute or so with a timestamp
    of the last time that each thread was running. If you ever want to know
    when a thread crashed (or if a thread has crashed) just check the
    logs/threads_running.json file
    """
    # if the logs/threads_running.json file doesn't exist, create it
    create_if_not_exists(path, default_content="{}", delete_if_exists=False)
            
    while True:
        for thread in threads:
            # if the current thread is still running, then udpate the
            # logs/threads_running.json file with the current time
            if thread.is_alive():
                with open(path, "r+") as file:
                    running_dict = json.load(file)
                    running_dict[thread.name] = current_time(seconds=True)
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
        time.sleep(sleep_duration)
        
def run_threads(thread_dict, seconds_between_threads=3, monitor=True, monitor_path="logs/threads_running.json", monitor_sleep_duration=60):
    
    threads = []
    
    for thread_name, thread in thread_dict.items():
        t = threading.Thread(target=thread, name=thread_name)
        threads.append(t)
        t.start()
        time.sleep(seconds_between_threads)
        
    if monitor:
        t = threading.Thread(target=monitor_threads, args=(threads,), kwargs={'sleep_duration': 1}, name="monitor_threads")
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
def reimport(statements, globals):
    """ takes in python import code represented as a string or a list of strings, and
    it reimports all of the modules and functions in the import statements. This allows
    you to reimport modules that have been changed in the background without having to
    restart the kernel.
    
    Example Call: reimport("import numpy as np", globals())
    
    The input can either be:
    - a string with a single import statement
      ex: 'import numpy as np' or 'from random import choice, randint'
    - a multiline string with multiple import statements
      ex: '''import numpy as np
      import pandas as pd
      from random import choice, randint'''
    - a list of strings where each string is an import statement
      ex: ['import numpy as np', 'import pandas as pd', 'from random import choice, randint']
      
    This also handles situations where imports look like:
    from config import (
        thing1,
        thing2,
        thing3
    )
    
    or that have the line continuation character \\ at the end of a line:
    from config import thing 1, \\
        thing2, \\
        thing3
    
    
    raises:
    AttributeError: if the module or function is not found in sys.modules
    
    GLOBALS NOTE:
    Note that the function call requires that you pass in globals() as the second argument
    so that the function can update the global variables with the new imports. If you don't
    pass in globals(), then the function will not be able to update the global variables with
    the new imports.
    
    DEPENDENCY NOTE: 
    If you make changes to two files file_1.py and file_2.py, and file_2 imports file_1, if you
    reimport file_2, then the changes in file_1 will not be reflected in file_2. You must reimport
    both file_1 and file_2 to see the changes in file_1 reflected in file_2.
       
    """
    ##########################
    # inner helper functions #
    ##########################

    def _import_one(module:str):
        """ takes in a string like 'random' or 'numpy as np' and imports that
        module with its alias if given
        """
        if ' as ' in module:
            module_name, alias = module.split(' as ')
            module_name = module_name.strip()
            alias = alias.strip()
        else:
            module_name, alias = module, None
            
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
            module_obj = sys.modules[module_name]
        else:
            if "." in module_name:
                module_obj = _import_module_hierarchy(module_name)[module_name]
            else:
                module_obj = importlib.import_module(module_name)
        globals[module_name] = module_obj
        
        if alias:
            globals[alias] = module_obj
            
        return module_obj
            
    def _import_sub_func(module, func):
        """ handles something like "from random import randint" where 'random'
        is the module and 'randint' is the function
        """
        if module not in sys.modules:
            raise AttributeError(f"Module {module} not found in sys.modules")
        module_obj = sys.modules[module]
        
        if ' as ' in func:
            func_name, alias = func.split(' as ')
            func_name = func_name.strip()
            alias = alias.strip()
        else:
            func_name, alias = func.strip(), None
        
        if func_name in dir(module_obj):
            if alias:
                globals[alias] = getattr(module_obj, func_name)
            else:
                globals[func_name] = getattr(module_obj, func_name)
        else:
            raise AttributeError(f"Function {func_name} not found in module {module}")
        
    def _get_sub_modules(package_path:str):
        """ takes in a package_path like directory.sub_directory and returns a 
        string of all of the modules in that directory like ['directory.sub_directory.module1',
        'directory.sub_directory.module2', ...]
        """
        module_obj = _import_one(package_path)
        if not hasattr(module_obj, "__path__"):
            return []
        
        return [
            module_info.name
            for module_info in pkgutil.walk_packages(module_obj.__path__, module_obj.__name__ + ".")
        ]

    def _get_type_as_str(parent_module, name):
        """ returns 'module' if the name is a module/package, 'function' if the name
        is a function inside the parent_module.py file, and '*' if the name is a wildcard
        import like 'from module import *'
        """
        if name.strip() == "*":
            return "*"
        
        if isinstance(parent_module, str):
            parent_module_obj = sys.modules[parent_module]
        else:
            parent_module_obj = parent_module
        
        # if there is an alias in the name, remove it
        main_portion = name.split(" as ")[0].strip()
        
        # packages with modules inside of them don't have attributes for each of the
        # modules inside of them. Modules with functions, do have string attributes
        # for each of their functions
        full_path = f"{parent_module_obj.__name__}.{main_portion}"
        if full_path in _get_sub_modules(parent_module):
            return "module"
        else:
            return "function"
        
    def _import_all_sub_funcs(module):
        """ handles a situation like from module_name import * where it imports
        all of the functions inside the module_name.py file """
        if module not in sys.modules:
            raise AttributeError(f"Module {module} not found in sys.modules")
        module_obj = sys.modules[module]
        
        for func_name in dir(module_obj):
            if not func_name.startswith("_"):
                globals[func_name] = getattr(module_obj, func_name)

    def _import_sub_module_component(parent_module, sub_component):
        """ handles a situation like 'from parent_module import sub_component' whether
        that sub_component is another package/module, function or the '*' wildcard
        """
        sub_component_type = _get_type_as_str(parent_module, sub_component)
        if sub_component_type == 'module':
            full_module_name = f"{parent_module}.{sub_component}"
            _import_one(full_module_name)
        elif sub_component_type == 'function':
            _import_sub_func(parent_module, sub_component)
        elif sub_component_type == '*':
            _import_all_sub_funcs(parent_module)
        else:
            print(f"Unknown type: {sub_component_type}")
            
    def _import_module_hierarchy(module_path:str):
        """ This takes in a module_path like 'directory.sub_directory.module_name'
        and imports all the modules in the hierarchy like directory, 
        director.sub_directory, director.sub_directory.module_name
        
        returns a dictionary where the keys are the module paths and the values 
        are the module objects
        """
        parts = module_path.split(".")
        
        # keep a record of the module objects for later use
        module_objs = {}
        
        curr_module = ""
        for i in range(len(parts)):
            # import everything up to the current module
            curr_module = ".".join(parts[:i+1])
            if curr_module in sys.modules:
                importlib.reload(sys.modules[curr_module])
                module_obj = sys.modules[curr_module]
            else:
                module_obj = importlib.import_module(curr_module)
            # add the current_module path to the module_objs dictionary
            module_objs[curr_module] = module_obj
            globals[curr_module] = module_obj
            
        return module_objs
            
    def _process_single_import_statement(import_statement:str):
        """ takes in a single import statement like 'import numpy as np' or 'from random import choice'
        and imports the module or function
        """
        # remove whitespace at the start and end of the import statement
        import_statement = import_statement.strip()
        
        # extract all of the modules names ('os', 'numpy as np', 'jeffutils.utils', 'stack_trace')
        module_re = re.compile(
            r"("
                r"\b(?!(?:import|from|as))[a-zA-Z0-9._]+\b\sas\s\b(?!(?:import|from|as))[a-zA-Z0-9._]+\b|"
                r"\b(?!(?:import|from|as))[a-zA-Z0-9._]+\b|"
                r"(?!(?:import))[*]"
            r")")
        module_names = module_re.findall(import_statement)
        
        # if a vanilla import statement, import each module
        if import_statement.startswith("import"):
            for module_name in module_names:
                _import_one(module_name)
            
        # if importing functions/sub-modules from a module
        elif import_statement.startswith("from"):
            
            # import the base module
            module = module_names[0]
            _import_one(module)
            
            # import each sub-function from the module
            for sub_component in module_names[1:]:
                _import_sub_module_component(module, sub_component)
                
    def _get_statements_from_str(statements):
        """ takes in a statement representing python import code and returns a list of
        all of the import statements in that code. This handles situations where the import
        statements are split by newlines or have (...) blocks in them
        """
        # first strip the input string of any leading/trailing whitespace
        statements = statements.strip()
        
        # if there is any () to break up an import statement, convert this to one line
        if "(" in statements and ")" in statements:
            
            # extract the (...) block
            start_paren = statements.index("(")
            end_paren = statements.index(")")
            sub_str = statements[start_paren:end_paren+1]
            
            # make the entire statement a single line
            sub_strs = sub_str.split("\n")
            sub_strs = [s.strip() for s in sub_strs]
            sub_str = " ".join(sub_strs)
            
            # get rid of the parentheses
            sub_str = sub_str.replace("(", " ").replace(")", " ")
            
            # replace the original statement with the new one and call the function again recursively
            new_statements = statements[:start_paren] + sub_str + statements[end_paren+1:]
            return _get_statements_from_str(new_statements)
        
        # if any '\' python line continuation characters are present, remove them
        if "\\" in statements:
            slash_remove = re.compile(r"\\\s?\n\s?")
            statements = slash_remove.sub("", statements)
        
        # initialize a list of all of the statements that will be processed
        statements_res = []
        
        # if there are multiple import statements split by newlines, split them up
        if "\n" in statements:
            statements = statements.split("\n")
            for statement in statements:
                statements_res.append(statement)
        else:
            statements_res.append(statements)
        
        return statements_res
    
    ##########################
    # main code for reimport #
    ##########################
    
    if isinstance(statements, str):
        for statement in _get_statements_from_str(statements):
            _process_single_import_statement(statement)
    elif isinstance(statements, list) and len(statements) > 0 and isinstance(statements[0], str):
        for statement in statements:
            for sttmnt in _get_statements_from_str(statement):
                _process_single_import_statement(sttmnt)
    else:
        raise ValueError(
            "Input should be a string or a list of strings, where each string is "
            "an import statement like 'import random', 'import numpy as np', "
            "'from jeffuitls.utils import stack_trace', etc."
        )


def create_if_not_exists(file_path, default_content="", delete_if_exists=False):
    """ if the file_path doesn't exist, it creates the directories
    and the file with the default_content.
    
    returns the file_path
    """
    # cehck to see if the file_path provided is a directory or a file
    file_extension = os.path.splitext(file_path)[1]
    is_directory = file_extension == ""
    
    # if the file_path is a directory, add a trailing slash
    if is_directory and file_path[-1] != "/":
        file_path += "/"
    
    # create the directory if it doesn't exist
    dir_name = os.path.dirname(file_path)
    if dir_name != "" and not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    # if the user wants to delete the file and replace it with the default content
    # and the file exists, delete it
    if delete_if_exists and os.path.exists(file_path):
        os.remove(file_path)
        
    # if the file_path is not a directory, and the file doesn't exist,
    # make it and fill it with the default content
    if not is_directory and not os.path.exists(file_path):
        with open(file_path, "w+") as f:
            f.write(default_content)
            f.close()
    
    return file_path

############################################################
#                           COLOR                          #
############################################################

# ANSI escape codes for text color
class Color:
    """ Valid colors:
    RESET
    BLACK
    RED
    GREEN
    YELLOW
    BLUE
    MAGENTA
    CYAN
    WHITE
    ORANGE
    
    BRIGHT_GREEN
    BRIGHT_CYAN
    BRIGHT_YELLOW
    BRIGHT_RED
    BRIGHT_BLACK
    BRIGHT_BLUE
    BRIGHT_MAGENTA
    BRIGHT_WHITE
    BRIGHT_ORANGE

    valid background colors
    
    BG_BLACK
    BG_RED
    BG_GREEN
    BG_YELLOW
    BG_BLUE
    BG_MAGENTA
    BG_CYAN
    BG_WHITE

    BG_BLACK_FAINT
    BG_WHITE_FAINT
        
    BG_BRIGHT_BLACK
    BG_BRIGHT_RED
    BG_BRIGHT_GREEN
    BG_BRIGHT_YELLOW
    BG_BRIGHT_BLUE
    BG_BRIGHT_MAGENTA
    BG_BRIGHT_CYAN
    BG_BRIGHT_WHITE
    """
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    ORANGE = "\033[38;5;208m"

    BRIGHT_GREEN = "\u001b[32;1m"
    BRIGHT_CYAN = "\u001b[36;1m"
    BRIGHT_YELLOW = "\u001b[33;1m"
    BRIGHT_RED = "\u001b[31;1m"
    BRIGHT_BLACK = "\u001b[30;1m"
    BRIGHT_BLUE = "\u001b[34;1m"
    BRIGHT_MAGENTA = "\u001b[35;1m"
    BRIGHT_WHITE = "\u001b[37;1m"
    BRIGHT_ORANGE = "\u001b[38;5;208;1m"

    BG_BLACK = "\u001b[40m"
    BG_RED = "\u001b[41m"
    BG_GREEN = "\u001b[42m"
    BG_YELLOW = "\u001b[43m"
    BG_BLUE = "\u001b[44m"
    BG_MAGENTA = "\u001b[45m"
    BG_CYAN = "\u001b[46m"
    BG_WHITE = "\u001b[47m"
    
    BG_BLACK_FAINT = "\u001b[48;5;239"
    BG_WHITE_FAINT = "\u001b[48;5;241"

    BG_BRIGHT_BLACK = "\u001b[40;1m"
    BG_BRIGHT_RED = "\u001b[41;1m"
    BG_BRIGHT_GREEN = "\u001b[42;1m"
    BG_BRIGHT_YELLOW = "\u001b[43;1m"
    BG_BRIGHT_BLUE = "\u001b[44;1m"
    BG_BRIGHT_MAGENTA = "\u001b[45;1m"
    BG_BRIGHT_CYAN = "\u001b[46;1m"
    BG_BRIGHT_WHITE = "\u001b[47;1m"


def print_colored(text, color):
    """prints the text string int he specified color"""
    print(color + text + Color.RESET)


def col(text, color):
    """returns a printable string in a specified color"""
    return color + text + Color.RESET


"""
Example:
print_colored("Hello World!", Color.BRIGHT_GREEN) # prints "Hello World!" in bright green
colored_trext = col("Hello World!", Color.BRIGHT_GREEN) # "Hello World!" in bright green
"""

