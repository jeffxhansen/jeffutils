# jeffutils

Welcome to Jeff Hansen's suite of useful python functions! I use lots of these functions on most of my Data Analysis, Backend-Dev, and Machine Learning projects, and I hope you find some of them useful as well!

# Installation

You can install this package from `PyPi` with
```
pip install jeffutils
```

# Getting Started

At the top of your python file, you can import any function from the `jeffutils.util` module

```python
from constants import EPOCHS, model
from ml import train, test, validate
from jeffutils.utils import reimport, stack_trace, log_print
```

# Documentation

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
