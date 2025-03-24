import abc
import threading

from typing import List

class ApplyFunction(abc.ABC):
    @abc.abstractmethod
    def apply(self, func: callable, lst: List) -> List:
        raise NotImplementedError

class ApplyFunctionThread(ApplyFunction):

    __metaClasses__ = (abc.ABCMeta)
    __abstractmethods__ = {'apply'}

    def apply(self, func: callable, lst: List) -> List:
        def worker(func, lst, result):
            result.append(func(lst))
        
        result = []
        threads = []

        if not lst:
            return result

        for i in range(len(lst)):
            t = threading.Thread(target=worker, args=(func, lst[i], result))
            threads.append(t)
            t.start()

        [t.join() for t in threads]

        return result