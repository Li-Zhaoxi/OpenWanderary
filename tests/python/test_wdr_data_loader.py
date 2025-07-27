import common  # noqa
import pytest

from wanderary import DataLoader


def test_data_loader_register_names():
    regist_names = DataLoader.RegisteredNames()
    print(regist_names)
    dst_regist_names = set({"SimpleImageDataset"})
    assert regist_names == dst_regist_names, regist_names


if __name__ == "__main__":
    pytest.main(["-s", __file__])
