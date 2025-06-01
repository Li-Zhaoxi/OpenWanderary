#pragma once

#include <string>
#include <vector>

namespace wdr {

template <typename DType>
std::vector<DType> ReadBytesFromFile(const std::string &filename);

}

#include <wanderary/utils/impl/file_io.hpp>
