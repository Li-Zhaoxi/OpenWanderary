#pragma once
#include <string>

#include <glog/logging.h>
#include <wanderary/utils/path.h>
#include <wanderary/utils/yaml_utils.h>

namespace wdr {

template <typename DataType>
DataType LoadYaml(const std::string &filepath) {
  CHECK(wdr::path::exist(filepath)) << "File " << filepath << " does not exist";
  YAML::Node node = YAML::LoadFile(filepath);
  if constexpr (std::is_same_v<DataType, wdr::json>) {
    return yaml2json(node);
  } else if constexpr (std::is_same_v<DataType, YAML::Node>) {
    return node;
  } else {
    LOG(FATAL) << "Unsupported type " << typeid(DataType).name();
  }
}

}  // namespace wdr
