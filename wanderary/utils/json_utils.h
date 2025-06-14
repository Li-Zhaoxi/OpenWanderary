#pragma once
#include <string>

#include <nlohmann/json.hpp>

namespace wdr::utils {

using json = nlohmann::json;

template <typename DType = json>
DType GetData(const json &data, const std::string &key,
              const std::optional<DType> &default_value = std::nullopt);

}  // namespace wdr::utils

#include <wanderary/utils/impl/json_utils.hpp>
