#include "wanderary/utils/yaml_utils.h"

#include <string>

#include <glog/logging.h>

namespace wdr {

namespace {
json convert_scalar(const Yaml &node) {
  int i = 0;
  double d = 0.0;
  bool b = false;
  std::string s = "";
  if (YAML::convert<int>::decode(node, i))
    return json(i);
  else if (YAML::convert<double>::decode(node, d))
    return json(d);
  else if (YAML::convert<bool>::decode(node, b))
    return json(b);
  else if (YAML::convert<std::string>::decode(node, s))
    return json(s);
  else
    LOG(FATAL) << "Cannot convert scalar: " << node;
}
}  // namespace

json yaml2json(const Yaml &yaml) {
  json res;
  switch (yaml.Type()) {
    case YAML::NodeType::Null:
      res = nullptr;
      break;
    case YAML::NodeType::Scalar:
      res = convert_scalar(yaml);
      break;
    case YAML::NodeType::Sequence:
      for (const auto &node : yaml) res.push_back(yaml2json(node));
      break;
    case YAML::NodeType::Map:
      for (const auto &node : yaml)
        res[node.first.as<std::string>()] = yaml2json(node.second);
      break;
    default:
      LOG(FATAL) << "Cannot convert node: " << yaml
                 << ", node type: " << static_cast<int>(yaml.Type());
      break;
  }
  return res;
}

}  // namespace wdr
