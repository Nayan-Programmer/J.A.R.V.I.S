INTENT_OPEN              = "open"
INTENT_PLAY              = "play"
INTENT_GOOGLE_SEARCH     = "google_search"
INTENT_YOUTUBE_SEARCH    = "youtube_search"
INTENT_GENERATE_IMAGE    = "generate_image"
INTENT_GENERATE_IMAGE_EMAIL = "generate_image_email"
INTENT_CONTENT           = "content"
INTENT_OPEN_WEBCAM       = "open_webcam"
INTENT_CLOSE_WEBCAM      = "close_webcam"
INTENT_GENERATE_PPT      = "generate_ppt"
INTENT_GENERATE_PPT_EMAIL = "generate_ppt_email"
INTENT_TRACK_PHONE       = "track_phone"
INTENT_GENERATE_CODE     = "generate_code"
INTENT_GENERATE_CODE_EMAIL = "generate_code_email"

ROUTE_TO_INTENT: dict[str, str] = {
    "open":                   INTENT_OPEN,
    "play":                   INTENT_PLAY,
    "google_search":          INTENT_GOOGLE_SEARCH,
    "youtube_search":         INTENT_YOUTUBE_SEARCH,
    "generate_image":         INTENT_GENERATE_IMAGE,
    "generate image":         INTENT_GENERATE_IMAGE,
    "generate_image_email":   INTENT_GENERATE_IMAGE_EMAIL,
    "generate image email":   INTENT_GENERATE_IMAGE_EMAIL,
    "content":                INTENT_CONTENT,
    "open_webcam":            INTENT_OPEN_WEBCAM,
    "close_webcam":           INTENT_CLOSE_WEBCAM,
    "generate_ppt":           INTENT_GENERATE_PPT,
    "generate ppt":           INTENT_GENERATE_PPT,
    "generate_ppt_email":     INTENT_GENERATE_PPT_EMAIL,
    "generate ppt email":     INTENT_GENERATE_PPT_EMAIL,
    "track_phone":            INTENT_TRACK_PHONE,
    "track phone":            INTENT_TRACK_PHONE,
    "generate_code":          INTENT_GENERATE_CODE,
    "generate code":          INTENT_GENERATE_CODE,
    "generate_code_email":    INTENT_GENERATE_CODE_EMAIL,
    "generate code email":    INTENT_GENERATE_CODE_EMAIL,
}

BACKGROUND_TASK_TYPES: set[str] = {
    INTENT_GENERATE_IMAGE,
    INTENT_GENERATE_IMAGE_EMAIL,
    INTENT_CONTENT,
    INTENT_GENERATE_PPT,
    INTENT_GENERATE_PPT_EMAIL,
    INTENT_TRACK_PHONE,
    INTENT_GENERATE_CODE,
    INTENT_GENERATE_CODE_EMAIL,
}

IMMEDIATE_TASK_TYPES: set[str] = {
    INTENT_OPEN,
    INTENT_PLAY,
    INTENT_GOOGLE_SEARCH,
    INTENT_YOUTUBE_SEARCH,
    INTENT_OPEN_WEBCAM,
    INTENT_CLOSE_WEBCAM,
}
