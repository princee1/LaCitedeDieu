
from enum import Enum
from functools import wraps,lru_cache
from typing import Any, Literal, Callable
from json import dump
from inspect import signature
from time import perf_counter,time
import psutil,os



############################################      CONSTANT        ###########################################
MAX_TIME = 60*15
MAX_RAM =   4_294_967_296/2


############################################      Utils        ###########################################


def hash_args(args):
    a, k = args
    return str(a)+"<==>"+str(k)


############################################       Seeker        ###########################################

class SeekerResultMode(Enum):
    """Enumeration for specifying the result output mode of the Seeker."""
    PRINT = 'print'
    JSON = 'json'


class Seeker:
    """
    A class that tracks the execution time and call count of a decorated function.

    Attributes:
        mode (SeekerResultMode): The output mode for results (e.g., print or JSON).
        data (dict): A dictionary storing execution time and call count for each unique argument.
        total_time (float): The total execution time for all calls to the tracked function.
        calling_counter (int): The total number of times the function has been called.
        parameter (dict): A dictionary of the function's parameters.

    Methods:
        init(func): Initializes the Seeker with the specified function.
        add(args, time, result=None): Records the execution time and updates the call count.
        __getitem__(args): Retrieves data for the specified arguments.
        __delitem__(args): Deletes the data entry for the specified arguments.
        __call__(args, kwds): Placeholder for callable functionality.
        order(key): Orders the recorded data based on time or count.
        max(key): Returns the maximum recorded entry based on the specified key.
        min(key): Returns the minimum recorded entry based on the specified key.
    """

    TIME_KEY = 'time'
    COUNT_KEY = 'count'
    RESULT_KEY = 'result'

    Cache: dict[str, 'Seeker'] = {}

    def __init__(self, mode: SeekerResultMode = SeekerResultMode.JSON) -> None:
        """
        Initializes the Seeker with the specified output mode.

        Args:
            mode (SeekerResultMode): The mode for outputting results (default is JSON).
        """
        self.mode = mode

    def init(self, func: Callable) -> None:
        """
        Initializes the Seeker with the given function.

        Args:
            func (Callable): The function to be tracked.
        """
        self.seek_name = func.__name__
        self.func = func
        self.data: dict[Any, dict[str, int | float]] = {}
        self.total_time = 0
        self.calling_counter = 0
        self.parameter = signature(func).parameters
        Seeker.Cache[self.seek_name] = self

    def add(self, args, time, result=None):
        """
        Records the execution time and updates the call count for the given arguments.

        Args:
            args (tuple): The arguments used in the function call.
            time (float): The time taken for the function to execute.
            result (Any, optional): The result of the function call (default is None).
        """
        args = hash_args(args)
        self.calling_counter += 1
        self.total_time += time
        if args not in self.data:
            self.data[args] = {
                Seeker.TIME_KEY: time,
                Seeker.COUNT_KEY: 1,
            }
            return
        self.data[args][Seeker.COUNT_KEY] += 1
        self.data[args][Seeker.TIME_KEY] += time

    def __getitem__(self, args):
        """
        Retrieves the execution data for the specified arguments.

        Args:
            args (tuple): The arguments used in the function call.

        Returns:
            dict: The recorded data for the specified arguments.
        """
        args = hash_args(args)
        return self.data[args]

    def __delitem__(self, args):
        """
        Deletes the recorded data entry for the specified arguments.

        Args:
            args (tuple): The arguments used in the function call.
        """
        args = hash_args(args)
        self.data.pop(args)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Placeholder method to make Seeker callable."""
        pass

    def __len__(self) -> int:
        """Returns the number of distinct calls recorded."""
        return len(self.data)

    def __iter__(self):
        """Returns an iterator over the recorded data."""
        return iter(self.data)

    def __contains__(self, args) -> bool:
        """
        Checks if the specified arguments have recorded data.

        Args:
            args (tuple): The arguments to check.

        Returns:
            bool: True if data exists for the specified arguments, False otherwise.
        """
        return hash_args(args) in self.data

    @property
    def reuse_ratio(self) -> float:
        """
        Calculates the reuse ratio of the function calls.

        Returns:
            float: The percentage of calls that reused cached results.
        """
        return 100 - (self.__len__() * 100 / self.calling_counter)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Seeker object.

        Returns:
            str: The representation including function name, call count, total time, and reuse ratio.
        """
        try:
            return (f'Seeker(Function: {self.func}, Name: {self.seek_name},\n'
                    f'\t\tCalling_Counter: {self.calling_counter}, Total_Time: {self.total_time}, '
                    f'Reuse_Ratio: {self.reuse_ratio:0.4f}%, Distinct_Call: {self.__len__()} )')
        except ZeroDivisionError:
            return 'Target function has not been called yet'

    def __str__(self) -> str:
        """
        Returns a string summary of the Seeker object, including saved data.

        Returns:
            str: Summary including potential time saved and data saved to file.
        """
        try:
            print(f'Time Potentially Saved: {(self.reuse_ratio / 100) * self.total_time}')
            dump({"parameter": str(self.parameter), "data": self.data}, open(self.seek_name + ".json", 'w'))
            print(f'{self.func} Data Saved')
            return self.__repr__()
        except ZeroDivisionError:
            return 'Target function has not been called yet'
        except AttributeError:
            return 'Object has not been initialized... Make sure to put in the Seek decorator'

    def __add__(self, other: object):
        """Placeholder method for adding two Seekers together."""
        pass

    def order(self, key: Literal['time', 'count']) -> list:
        """
        Orders the recorded data based on the specified key.

        Args:
            key (Literal['time', 'count']): The key to order by ('time' or 'count').

        Returns:
            list: A sorted list of items based on the specified key.
        """
        return sorted(self.data.items(), key=lambda item: item[1][key])

    def max(self, key: Literal['time', 'count'] = 'count') -> str:
        """
        Retrieves the maximum recorded entry based on the specified key.

        Args:
            key (Literal['time', 'count']): The key to check for maximum (default is 'count').

        Returns:
            str: A string representation of the maximum entry.
        """
        return f'Max({key}): {self.order(key)[-1]}'

    def min(self, key: Literal['time', 'count'] = 'count') -> str:
        """
        Retrieves the minimum recorded entry based on the specified key.

        Args:
            key (Literal['time', 'count']): The key to check for minimum (default is 'count').

        Returns:
            str: A string representation of the minimum entry.
        """
        return f'Min({key}): {self.order(key)[0]}'


def Seek(seeker: Seeker):
    """
    A decorator that tracks the execution time of a function and records it using a specified seeker.

    Args:
        seeker (Seeker): An instance of the Seeker class that handles the initialization and recording of execution times.

    Returns:
        Callable: A decorator that wraps the specified function to add timing functionality.

    The decorator initializes the seeker with the target function and, upon each call,
    measures the time taken for execution. It then records the timing information in the
    provided seeker instance.

    Example:
        @Seek(Seeker())
        def example_function(x, y):
            # Some time-consuming computations
            return x + y
    """
    def decorator(func: Callable):
        seeker.init(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = func(*args, **kwargs)
            end_time = perf_counter()
            seeker.add((args, kwargs), end_time - start_time)
            return result
        
        return wrapper

    return decorator

##############################################   Time             ###########################################


def TimeW_Args(func: Callable):
    """
    A decorator that measures the execution time of a function.

    Args:
        func (Callable): The function to be decorated and timed.

    Returns:
        Callable: A wrapper function that calls the original function and prints its execution time.

    The decorator prints the time taken by the function to execute, along with the function's name
    and a hash of the arguments passed to it.

    Example:
    >>> \n\t@Time
        def example_function(x, y):
            # Some time-consuming computations
            return x + y
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        print(
            f'Func: {func.__name__}\nArgs: {hash_args((args,kwargs))}\nTime: {end_time-start_time} sec')
        return result

    return wrapper


def Time(func: Callable):
    """
    A decorator that measures the execution time of a function.

    Args:
        func (Callable): The function to be decorated and timed.

    Returns:
        Callable: A wrapper function that calls the original function and prints its execution time.

    The decorator prints the time taken by the function to execute, along with the function's name
    and a hash of the arguments passed to it.

    Example:
    >>> \n\t@Time
        def example_function(x, y):
            # Some time-consuming computations
            return x + y
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        print(
            f'Func: {func.__name__} Time: {end_time-start_time} sec')
        return result

    return wrapper

##############################################  Memoirization     ###########################################


def Memoization(func: Callable):
    """
    A decorator that caches the results of a function to optimize performance for
    repeated calls with the same arguments.

    Args:
        func (Callable): The function to be decorated and memoized.

    Returns:
        Callable: A wrapper function that returns cached results for previously
        computed arguments, or computes and caches the result if not previously cached.

    This decorator significantly speeds up function execution by storing previously
    computed results and returning them directly when the same inputs are encountered.

    Example:
    >>> \n\t@Memoization
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

    >>>    print(fibonacci(10))  # This will compute only once for each unique n.
    """

    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = hash_args((args, kwargs))
        if key not in cache:
            cache[key]= func(*args, **kwargs)
        return cache[key]
    return wrapper


##############################################   Monitor      ###########################################


def Monitor(func:Callable):

    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss
        start_cpu = process.cpu_percent(interval=None)
        
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        
        end_mem = process.memory_info().rss
        end_cpu = process.cpu_percent(interval=None)

        used_mem = abs(end_mem-start_mem)
        computed_time = end_time - start_time
        print('*'*30)
        print(f"Memory usage: {used_mem} bytes")
        print(f"CPU usage: {(end_cpu - start_cpu)}%")
        print(f"Execution time: {computed_time:.4f} seconds")
        print(f'Environnement: {used_mem*100/MAX_RAM:.4f} RAM USED % -',f'{computed_time*100/MAX_TIME:.4f} TIME USED %' )
        print('*'*30)
        
        return result
    return wrapper
