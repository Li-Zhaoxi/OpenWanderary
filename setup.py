import os
import shutil

import setuptools
import yaml

# python ../setup.py clean --all
# python ../setup.py bdist_wheel


def clear_package(package_path):
    if os.path.exists(package_path):
        shutil.rmtree(package_path)
    if os.path.exists("dist"):
        shutil.rmtree("dist")


def group_package(package_name, pybind_paths):
    # 配置包根目录
    package_root = os.path.join("package", package_name)
    clear_package(package_root)
    os.makedirs(package_root, exist_ok=True)

    # 拷贝pybind模块
    pymod_paths = []
    for pybind_path in pybind_paths:
        dstpath = os.path.join(package_root, os.path.basename(pybind_path))
        shutil.copy2(pybind_path, dstpath)
        pymod_paths.append(dstpath)

    # 构造Init文件
    init_path = os.path.join(package_root, "__init__.py")
    with open(init_path, "w") as f:
        for pybind_path in pybind_paths:
            package_name = os.path.basename(pybind_path).split(".")[0]
            f.write(f"from .{package_name} import *\n")

    # 构造MANIFEST.in文件
    manifest_path = "MANIFEST.in"
    with open(manifest_path, "w") as f:
        f.write("include {}\n".format(" ".join(pymod_paths)))
        f.write("global-exclude *.pyc\n")


with open("py-config.yaml", "r") as f:
    py_config = yaml.safe_load(f)

with open(py_config["pybind_module_path"], "r") as f:
    pybind_paths = f.readlines()
    pybind_paths = [path.strip() for path in pybind_paths]


group_package(py_config["pakcage_name"], pybind_paths)

setuptools.setup(
    name=py_config["pakcage_name"],
    version=py_config["version"],
    url="https://github.com/Li-Zhaoxi/OpenWanderary",
    package_dir={"": "package"},
    packages=setuptools.find_packages("package"),
    python_requires='>=3.10, <3.11',
    include_package_data=True,
)
