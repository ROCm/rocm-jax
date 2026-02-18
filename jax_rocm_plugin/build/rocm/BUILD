load("@bazel_skylib//rules:common_settings.bzl", "string_flag")

string_flag(
    name = "sanitizer",
    build_setting_default = "none",
    values = [
        "none",
        "asan",
        "tsan",
    ],
)

config_setting(
    name = "asan",
    flag_values = {"//build/rocm:sanitizer": "asan"},
)

config_setting(
    name = "tsan",
    flag_values = {"//build/rocm:sanitizer": "tsan"},
)

filegroup(
    name = "sanitizer_ignore_lists",
    srcs = select({
        ":asan": [
            "asan_ignore_list.txt",
            "lsan_ignore_list.txt",
        ],
        ":tsan": ["tsan_ignore_list.txt"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "sanitizer_wrapper",
    srcs = ["sanitizer_wrapper.sh"],
    data = [":sanitizer_ignore_lists"],
    visibility = ["//visibility:public"],
)
