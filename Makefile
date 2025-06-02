
pip_source = https://pypi.tuna.tsinghua.edu.cn/simple

rely:
	set -ex; \
	pip3 install -r requirements.txt -i $(pip_source); \
	pre-commit install; \
	bash build_3rdpary.sh;

debug:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug; \
	make -j6 -C build/;

release:
	cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release; \
	make -j6 -C build_release/;
