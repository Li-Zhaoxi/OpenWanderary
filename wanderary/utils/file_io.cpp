#include "wanderary/utils/file_io.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

namespace wdr {

namespace {

// 移除字符串中的换行符
std::string remove_crlf(const std::string& str) {
  std::string result = str;
  result.erase(std::remove(result.begin(), result.end(), '\r'), result.end());
  result.erase(std::remove(result.begin(), result.end(), '\n'), result.end());
  return result;
}

}  // namespace

std::vector<std::string> ReadLinesFromFile(const std::string& filename) {
  std::ifstream infile(filename);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty()) continue;
    lines.push_back(remove_crlf(line));
  }
  return lines;
}

}  // namespace wdr
