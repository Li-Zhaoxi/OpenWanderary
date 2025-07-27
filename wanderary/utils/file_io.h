#pragma once

#include <string>
#include <vector>

namespace wdr {

template <typename DType>
std::vector<DType> ReadBytesFromFile(const std::string &filename);

std::vector<std::string> ReadLinesFromFile(const std::string &filename);

}  // namespace wdr

#include <wanderary/utils/impl/file_io.hpp>
