# content-aware-encoding
Processing 

static const AVOption cae_options[] = {
    { "alpha_complexity", "Threshold multiplier for complexity", OFFSET(alpha_complexity), AV_OPT_TYPE_DOUBLE, {.dbl = 0.5}, 0, 10, FLAGS },
    { "alpha_ssim", "Threshold multiplier for SSIM", OFFSET(alpha_ssim), AV_OPT_TYPE_DOUBLE, {.dbl = 0.8}, 0, 10, FLAGS },
    { "alpha_hist", "Threshold multiplier for Histogram Difference", OFFSET(alpha_hist), AV_OPT_TYPE_DOUBLE, {.dbl = 0.5}, 0, 10, FLAGS },
    { "alpha_dct", "Threshold multiplier for DCT Energy", OFFSET(alpha_dct), AV_OPT_TYPE_DOUBLE, {.dbl = 0.5}, 0, 10, FLAGS },
    { "alpha_sobel", "Threshold multiplier for Sobel Energy", OFFSET(alpha_sobel), AV_OPT_TYPE_DOUBLE, {.dbl = 0.5}, 0, 10, FLAGS },
    { "window_size", "Number of frames in the sliding window", OFFSET(window_size), AV_OPT_TYPE_INT, {.i64 = 30}, 1, 100, FLAGS },
    { "threshold_mode", "Thresholding mode (0: Median + alpha*MAD)", OFFSET(threshold_mode), AV_OPT_TYPE_INT, {.i64 = 0}, 0, 0, FLAGS },
    { "cooldown_frames", "Number of frames to ignore after a scene change", OFFSET(cooldown_frames), AV_OPT_TYPE_INT, {.i64 = 10}, 0, 100, FLAGS },
    { "required_consecutive_changes", "Number of consecutive detections to confirm scene change", OFFSET(required_consecutive_changes), AV_OPT_TYPE_INT, {.i64 = 2}, 1, 10, FLAGS },
    { "k_threshold", "Multiplier for MAD in adaptive threshold calculation", OFFSET(k_threshold), AV_OPT_TYPE_DOUBLE, {.dbl = 3.0}, 1.0, 10.0, FLAGS },
    { "max_weight", "Maximum sum of metric weights to prevent domination", OFFSET(max_weight), AV_OPT_TYPE_DOUBLE, {.dbl = 10.0}, 1.0, 100.0, FLAGS },
    { "frame_interval", "Number of frames between each scene change check", OFFSET(frame_interval), AV_OPT_TYPE_INT, {.i64 = 2}, 1, 1000, FLAGS },
    { NULL }
};

ffmpeg  -loglevel debug  -i /Users/osman/projects/video_encoding/Samples/video2.mp4 -vf  "cae" -f null -


[Parsed_cae_0 @ 0x6000027f2940] SAD_NEON: 1.688 ms
[Parsed_cae_0 @ 0x6000027f2940] HIST_NEON: 1.693 ms
[Parsed_cae_0 @ 0x6000027f2940] HIST_DIFF_NEON: 0.001 ms
[Parsed_cae_0 @ 0x6000027f2940] SSIM_NEON: 2.349 ms
[Parsed_cae_0 @ 0x6000027f2940] DCT: 4.154 ms
[Parsed_cae_0 @ 0x6000027f2940] SOBEL: 5.353 ms
[Parsed_cae_0 @ 0x6000027f2940] ENTROPY: 2.211 ms
[Parsed_cae_0 @ 0x6000027f2940] COLOR_VARIANCE: 2.384 ms
[Parsed_cae_0 @ 0x6000027f2940] MAD_MEDIAN: 0.025 ms
[Parsed_cae_0 @ 0x6000027f2940] ADAPTIVE_THRESHOLD: 0.003 ms

[Parsed_cae_0 @ 0x6000027f2940] GLOBAL: 21.313 ms

