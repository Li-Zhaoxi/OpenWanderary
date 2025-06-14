#pragma once

#include <map>
#include <set>

namespace wdr {

template <typename KeyType, typename ValueType>
bool contains(const std::map<KeyType, ValueType>& map, const KeyType& key) {
  return map.find(key) != map.end();
}

template <typename KeyType>
bool contains(const std::set<KeyType>& map, const KeyType& key) {
  return map.find(key) != map.end();
}

}  // namespace wdr
