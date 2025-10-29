#pragma once
#include <string>
#include <vector>

namespace wdr::path {

bool exist(const std::string &filepath, bool log = false);
std::string join(const std::vector<std::string> &filepaths);
std::string join(const std::string &path1, const std::string &name);
std::string dirname(const std::string &filepath);
std::string basename(const std::string &filepath);

// 文件后缀名
std::string extname(const std::string &filepath);

}  // namespace wdr::path
