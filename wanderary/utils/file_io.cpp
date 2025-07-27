#include "wanderary/utils/file_io.h"

#include <fstream>
#include <string>
#include <vector>

namespace wdr {

std::vector<std::string> ReadLinesFromFile(const std::string &filename) {
  std::ifstream infile(filename);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty()) continue;
    lines.push_back(line);
  }
  return lines;
}

}  // namespace wdr
