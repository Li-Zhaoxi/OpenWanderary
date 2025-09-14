#include "wanderary/utils/path.h"

#include <string>
#include <vector>

#include <glog/logging.h>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

namespace wdr::path {

bool exist(const std::string &filepath, bool log) {
  if (!fs::exists(filepath)) {
    if (log) LOG(INFO) << "File does not exist: " << filepath;
    return false;
  } else {
    return true;
  }
}

std::string join(const std::vector<std::string> &filepaths) {
  const int pathnum = filepaths.size();
  if (pathnum == 0) return "";

  fs::path basepath = filepaths[0];
  for (int idx = 1; idx < pathnum; idx++) basepath /= filepaths[idx];
  return basepath.string();
}

std::string join(const std::string &path1, const std::string &name) {
  fs::path basepath = path1;
  basepath /= name;
  return basepath.string();
}

std::string dirname(const std::string &filepath) {
  return fs::path(filepath).parent_path().string();
}

std::string basename(const std::string &filepath) {
  return fs::path(filepath).filename().string();
}

}  // namespace wdr::path
