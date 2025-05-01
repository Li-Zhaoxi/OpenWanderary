#include <glog/logging.h>
#include <gtest/gtest.h>

#include "wanderary/media/media_codec.h"

using CodecDescriptor = wdr::media::CodecDescriptor;

TEST(MediaCodec, GetCodecDescriptor) {
  const CodecDescriptor desc =
      wdr::media::GetCodecDescriptor(MediaCodecID::kMJPEG);
  EXPECT_TRUE(desc.has_value());

  if (!desc.has_value()) return;

  // mjpg id, 硬编码
  EXPECT_TRUE(desc.id() == MediaCodecID::kMJPEG)
      << "codec_id: " << MediaCodecID2str(desc.id());
  EXPECT_TRUE(desc.mode() == MediaCodecMode::kHARDWARE)
      << "codec_mode: " << MediaCodecMode2str(desc.mode());
  EXPECT_EQ(desc.name(), "mjpeg");
  EXPECT_EQ(desc.long_name(), "Motion JPEG");
  EXPECT_EQ(desc.mime_types(), "image/jpeg");
}
