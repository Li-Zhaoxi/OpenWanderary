cur_dir=$(pwd)

#
sudo apt-get install git-lfs libcli11-dev
git lfs install

# 安装nlohmann_json
cd ${cur_dir}/3rdparty/nlohmann_json
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DJSON_BuildTests=OFF
make -j6 -C build/
sudo make install -C build/

# 安装pybind
cd ${cur_dir}/3rdparty/pybind11
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPYBIND11_TEST=OFF
make -j6 -C build/
sudo make install -C build/
cd ${cur_dir}/3rdparty/pybind11_json
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
make -j6 -C build/
sudo make install -C build/

# 安装yamlcpp
cd ${cur_dir}/3rdparty/yaml-cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DYAML_CPP_BUILD_TOOLS=OFF -DBUILD_SHARED_LIBS=ON
make -j6 -C build/
sudo make install -C build/

# 安装indicators
cd ${cur_dir}/3rdparty/indicators
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
make -j6 -C build/
sudo make install -C build/

# 安装MCap
cd ${cur_dir}/3rdparty/mcap_builder
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
make -j6 -C build/
sudo make install -C build/
