from abc import ABC, abstractmethod


class BaseSearch(ABC):
    cache = {}

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def search(self, *args, **kwargs):
        pass

    @abstractmethod
    def get(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_models(self, *args, **kwargs):
        pass

    @abstractmethod
    def _process_query(self, *args, **kwargs):
        pass

    @abstractmethod
    def _process_advance(self, *args, **kwargs):
        pass
