# jeffutils

Welcome to Jeff Hansen's suite of useful python functions! I use lots of these functions on most of my Data Analysis, Backend-Dev, and Machine Learning projects, and I hope you find some of them useful as well!

# Installation

You can install this package from `PyPi` with
```
pip install jeffutils
```
or
```
python -m pip install --upgrade jeffutils
```

# Getting Started

At the top of your python file, you can import any function from the `jeffutils.util` module

```python
from constants import EPOCHS, model
from ml import train, test, validate
from jeffutils.utils import reimport, stack_trace, log_print
```

# Documentation

## Table of Conents

- [Useful python functionality](#useful-python-functionality)
    - [reimport](#reimport)
    - [stack_trace](#stack_trace)
    - [create_if_not_exists](#create_if_not_exists)
- [NumPy](#numpy)
    - README sections coming soon (code in src/jeffutils/utils)
- [Pandas](#pandas)
    - README sections coming soon (code in src/jeffutils/utils)
- [SQL](#sql)
    - README sections coming soon (code in src/jeffutils/utils)

# Useful Python Functionality

## reimport

This function allows you to reimport a python file without having to restart your session (if you are developing using a jupyter kernel)

For example, let's say you have this file structure
```Python
# constants.py
start = 0
end = 5
```
```Python
# file.py
from constants import start, end
from random import randint

def random_numbers(n):
    return [randint(start, end) for _ in range(n)]
```
Running this code looks like:
```Python
>>> from file import random_numbers
>>> random_numbers(10)
[3, 5, 2, 1, 3, 0, 4, 2, 0, 4]
```
Let's say you change the constants.py so that `start = 0` and `end=1`. If you try running the same code, you get:
```
>>> random_numbers(10)
[3, 5, 5, 4, 5, 0, 0, 5, 5, 5]
```
The python session is still using the old versions of the `start` and `end` variables. Without restarting your python session, you can run these lines of code:
```Python
>>> from jeffutils.utils import reimport
>>> reimport(["import constants", "from file import random_numbers"], globals())
>>> random_numbers(10)
[1, 1, 1, 0, 1, 1, 0, 0, 1, 1]
```
and you will see that the changes have been updated, without you having to restart the session and lose all other variables.

`reimport` also handles entire blocks of import statements like this:
```Python
from time import sleep
from constants import (
    long_var_name,
    another_long_var_name,
    EPOCHS
)
from file import func1, func2, func3
```
```Python
from jeffutils.utils import reimport
reimport("""from time import sleep
from constants import (
    long_var_name,
    another_long_var_name,
    EPOCHS
)
from file import func1, func2, func3""", globals())
```

## stack_trace

Are you tired of
```Python
try:
    # code here
except Exception as e:
    print(e)
```
not printing out the entire stack trace to help your debugging?

The `stack_trace` function takes in an exception, and it returns the entire stack_trace from that exception as a string, so you  can do something like
```Python
from jeffutils.utils import stack_trace
try:
    # code here
except Exception as e:
    print(stack_trace(e))
```

## create_if_not_exists

This function, `create_if_not_exists`, helps to ensure that a file or directory exists. If the specified path doesn't exist, the function will create the necessary directories and file with optional default content. Additionally, it can replace an existing file if specified.

```Python
from jeffutils.utils import create_if_not_exists
```

**Example 1: Creating a file and directory**

If both the `logs` directory and the `log.txt` file don't exist, calling the function with `"logs/log.txt"` will automatically create the directory `logs` and the file `log.txt`.

```Python
file_path = create_if_not_exists("logs/log.txt")
print(f"File created at: {file_path}") # -> "File created at: logs/log.txt"
```

**Example 2: Creating a file with default content**

If the `logs/threads_running.json` file doesn't exist, but the `logs/` directory does, calling this function with `create_if_not_exists("logs/threads_running.json", "{}")` will create the `threads_running.json` file and fill it with an empty dictionary `{}`.

```python
file_path = create_if_not_exists("logs/threads_running.json", "{}")
print(f"File created at: {file_path}") # -> "File created at: logs/threads_running.json"
```

# NumPy

Coming soon in the README. The code is in `src/jeffutils/utils.py` with docstrings and examples.

# Pandas

Coming soon in the README. The code is in `src/jeffutils/utils.py` with docstrings and examples.

# SQL

Coming soon in the README. The code is in `src/jeffutils/utils.py` with docstrings and examples.




