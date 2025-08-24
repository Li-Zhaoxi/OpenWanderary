#pragma once
#include <string>

#include <wanderary/utils/json_utils.h>
#include <yaml-cpp/yaml.h>

namespace wdr {

using Yaml = YAML::Node;

json yaml2json(const Yaml &yaml);

template <typename DataType>
DataType LoadYaml(const std::string &filepath);

}  // namespace wdr

#include <wanderary/utils/impl/yaml_utils.hpp>
