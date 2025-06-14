import common  # noqa
import pytest

from wanderary import ProcessManager


def test_process_register_names():
    regist_names = ProcessManager.RegisteredNames()
    print(regist_names)
    assert regist_names == set({"FormatImage"}), regist_names


if __name__ == "__main__":
    pytest.main(["-s", __file__])
