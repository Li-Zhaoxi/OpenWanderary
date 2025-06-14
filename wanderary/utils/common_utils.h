#pragma once

#include <map>

namespace wdr {

template <typename KeyType, typename ValueType>
bool contains(const std::map<KeyType, ValueType>& map, const KeyType& key) {
  return map.find(key) != map.end();
}

}  // namespace wdr
