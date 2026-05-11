from timelesseg.singleton import Singleton


class MappingObj:
    """
    This essentially allows you to do dict unpacking...
    """

    def __getitem__(self, item):
        return self.__dict__[item]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def get(self, __key, default):
        return self.__dict__.get(__key, default)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __contains__(self, o):
        return o in self.__dict__

    def update(self, otherdict):
        self.__dict__.update(otherdict)


class Config(MappingObj, metaclass=Singleton):
    pass