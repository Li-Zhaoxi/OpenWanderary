#include "wanderary/utils/common_utils.h"

#include <absl/crc/crc32c.h>

namespace wdr {

uint32_t ExtendAbslCRC32c(uint32_t init_crc, const char* buf, size_t size) {
  return static_cast<uint32_t>(absl::ExtendCrc32c(
      static_cast<absl::crc32c_t>(init_crc), absl::string_view(buf, size)));
}

}  // namespace wdr
