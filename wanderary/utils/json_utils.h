#pragma once
#include <string>

#include <nlohmann/json.hpp>

namespace wdr {

using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

template <typename DType = json>
DType GetData(const json &data, const std::string &key,
              const std::optional<DType> &default_value = std::nullopt);

void SaveJson(const std::string &filepath, const json &data);
void SaveJson(const std::string &filepath, const ordered_json &data);

json LoadJson(const std::string &filepath);

}  // namespace wdr

#include <wanderary/utils/impl/json_utils.hpp>
