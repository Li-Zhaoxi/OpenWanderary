#pragma once

#include <cstddef>
#include <cstdint>
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

// 计算字符串的 CRC32C 校验和
uint32_t ExtendAbslCRC32c(uint32_t init_crc, const char* buf, size_t size);

}  // namespace wdr
