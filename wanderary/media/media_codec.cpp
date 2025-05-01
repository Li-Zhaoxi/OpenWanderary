#include "wanderary/media/media_codec.h"

namespace wdr::media {

CodecDescriptor GetCodecDescriptor(MediaCodecID codec_id) {
  return CodecDescriptor(
      hb_mm_mc_get_descriptor(static_cast<media_codec_id_t>(codec_id)));
}

}  // namespace wdr::media
