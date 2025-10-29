#include <string>

#include <foxglove/CompressedImage.pb.h>
namespace wdr::msg {

void ConvertMsgImage(const std::string &image_path, int64_t timestamp,
                     const std::string &frame_id,
                     foxglove::CompressedImage *msg);

}  // namespace wdr::msg
