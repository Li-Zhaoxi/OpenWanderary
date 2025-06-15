import common  # noqa
import pytest

from wanderary import ProcessManager


def test_process_register_names():
    regist_names = ProcessManager.RegisteredNames()
    print(regist_names)
    dst_regist_names = set({"FormatImage", "ConvertYoloFeature"})
    assert regist_names == dst_regist_names, regist_names


if __name__ == "__main__":
    pytest.main(["-s", __file__])
