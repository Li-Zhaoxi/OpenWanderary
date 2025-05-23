#pragma once
#include <string>
#include <vector>

namespace wdr::path {

bool exist(const std::string &filepath, bool log = false);
std::string join(const std::vector<std::string> &filepaths);
std::string dirname(const std::string &filepath);

}  // namespace wdr::path
