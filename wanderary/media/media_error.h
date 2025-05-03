#pragma once
#include <hb_media_error.h>
#include <wanderary/utils/enum_traits.h>

ENUM_NUMBERED_REGISTER(
    MediaErrorCode,                   //
    ((kSuccess, 0, "media_success"))  //
    ((kUnknown, HB_MEDIA_ERR_UNKNOWN,
      "media_err_unknown"))  //
    ((kCodecNotFound, HB_MEDIA_ERR_CODEC_NOT_FOUND,
      "media_err_codec_not_found"))  //
    ((kCodecOpenFail, HB_MEDIA_ERR_CODEC_OPEN_FAIL,
      "media_err_codec_open_fail"))  //
    ((kCodecResponseTimeout, HB_MEDIA_ERR_CODEC_RESPONSE_TIMEOUT,
      "media_err_codec_response_timeout"))  //
    ((kCodecInitFail, HB_MEDIA_ERR_CODEC_INIT_FAIL,
      "media_err_codec_init_fail"))  //
    ((kOpNotAllowed, HB_MEDIA_ERR_OPERATION_NOT_ALLOWED,
      "media_err_operation_not_allowed"))  //
    ((kInsufficientRes, HB_MEDIA_ERR_INSUFFICIENT_RES,
      "media_err_insufficient_res"))  //
    ((kNoFreeInstance, HB_MEDIA_ERR_NO_FREE_INSTANCE,
      "media_err_no_free_instance"))  //
    ((kInvalidParams, HB_MEDIA_ERR_INVALID_PARAMS,
      "media_err_invalid_params"))  //
    ((kInvalidInstance, HB_MEDIA_ERR_INVALID_INSTANCE,
      "media_err_invalid_instance"))  //
    ((kInvalidBuffer, HB_MEDIA_ERR_INVALID_BUFFER,
      "media_err_invalid_buffer"))  //
    ((kInvalidCommand, HB_MEDIA_ERR_INVALID_COMMAND,
      "media_err_invalid_command"))                                        //
    ((kWaitTimeout, HB_MEDIA_ERR_WAIT_TIMEOUT, "media_err_wait_timeout"))  //
    ((kFileOperaFail, HB_MEDIA_ERR_FILE_OPERATION_FAILURE,
      "media_err_file_operation_failure"))  //
    ((kParamsSetFail, HB_MEDIA_ERR_PARAMS_SET_FAILURE,
      "media_err_params_set_failure"))  //
    ((kParamsGetFail, HB_MEDIA_ERR_PARAMS_GET_FAILURE,
      "media_err_params_get_failure"))                                        //
    ((kCodingFailed, HB_MEDIA_ERR_CODING_FAILED, "media_err_coding_failed"))  //
    ((kOutBufFull, HB_MEDIA_ERR_OUTPUT_BUF_FULL,
      "media_err_output_buf_full"))  //
    ((kUnsupportFeat, HB_MEDIA_ERR_UNSUPPORTED_FEATURE,
      "media_err_unsupported_feature"))  //
    ((kInvalidPriority, HB_MEDIA_ERR_INVALID_PRIORITY,
      "media_err_invalid_priority"))  //
)
ENUM_CONVERSION_REGISTER(MediaErrorCode, MediaErrorCode::kUnknown,
                         "media_err_unknown")
