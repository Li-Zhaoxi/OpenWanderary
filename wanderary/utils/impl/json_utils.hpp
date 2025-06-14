#pragma once
#include <string>

#include <wanderary/utils/json_utils.h>

namespace wdr::utils {

template <typename DType = json>
DType GetData(const json &data, const std::string &key,
              const std::optional<DType> &default_value) {
  if (data.contains(key)) {
    try {
      return data[key].get<DType>();
    } catch (const std::exception &e) {
      LOG(FATAL) << "Failed to read key: " << key << ", error: " << e.what()
                 << ", raw data: " << data[key].dump(2);
    }
  } else {
    if (default_value.has_value())
      return default_value.value();
    else
      LOG(FATAL) << "Key not found: " << key << ", raw data: " << data.dump();
  }
  return DType();  // compiler should not reach here
}

}  // namespace wdr::utils
