#pragma once
#include <fstream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <wanderary/utils/file_io.h>

namespace wdr {

template <typename DType>
std::vector<DType> ReadBytesFromFile(const std::string &filename) {
  std::vector<DType> res;

  std::ifstream infile(filename, std::ios::binary);
  CHECK(infile.is_open()) << "Failed to open file " << filename;

  infile.seekg(0, std::ios::end);
  const int64_t data_size = infile.tellg();

  const int data_num = data_size / sizeof(DType);
  CHECK_EQ(data_size, data_num * sizeof(DType))
      << "File size is not a multiple of " << sizeof(DType);

  res.resize(data_num);
  infile.seekg(0, std::ios::beg);
  infile.read(reinterpret_cast<char *>(res.data()), data_size);

  infile.close();

  return res;
}

}  // namespace wdr
