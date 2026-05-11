from dataclasses import dataclass

from timelesseg.singleton import Singleton
from timelesseg.configs.training import MappingObj, JoinedConfigs


@dataclass
class DummyConfig1(MappingObj, metaclass=Singleton):
    x: int = 1
    y: int = 2

@dataclass
class DummyConfig2(MappingObj, metaclass=Singleton):
    z: int = 3
    t: int = 4

@dataclass
class DummyConfig3(MappingObj, metaclass=Singleton):
    k: int = 3
    t: int = 4

def test():
    all_configs = JoinedConfigs(DummyConfig1(), DummyConfig2())
    assert all_configs.y == 2 and all_configs.z == 3
    try:
        all_configs = all_configs + DummyConfig3()
    except ValueError:
        print('Yayy')

    assert {**all_configs} == {'x': 1, 'y': 2, 'z': 3, 't': 4}, {**all_configs}