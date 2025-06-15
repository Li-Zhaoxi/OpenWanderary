#include "wanderary/utils/json_utils.h"

#include <fstream>
#include <string>
namespace wdr {

void SaveJson(const std::string &filepath, const json &data) {
  std::ofstream file(filepath);
  file << data.dump(2);
  file.close();
}

void SaveJson(const std::string &filepath, const ordered_json &data) {
  std::ofstream file(filepath);
  file << data.dump(2);
  file.close();
}

json LoadJson(const std::string &filepath) {
  std::ifstream file(filepath);
  json data;
  file >> data;
  file.close();
  return data;
}

}  // namespace wdr
