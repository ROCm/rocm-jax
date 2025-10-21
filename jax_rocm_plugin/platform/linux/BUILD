load("@rules_cc//cc:defs.bzl", "cc_toolchain", "cc_toolchain_suite")

package(default_visibility = ["//visibility:public"])

platform(
    name = "manylinux",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
        "@bazel_tools//tools/cpp:clang",
    ],
    exec_properties = {
        "container-image": "docker://938540721288.dkr.ecr.us-east-1.amazonaws.com/amd/rbe@sha256:48515bbef9780ce32bdadd6e236d344217009fec4ed47a7809f41707ef8b8551",
        "OSFamily": "Linux",
    },
)

platform(
    name = "manylinux_gpu",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
        "@bazel_tools//tools/cpp:clang",
    ],
    exec_properties = {
        "container-image": "docker://938540721288.dkr.ecr.us-east-1.amazonaws.com/amd/rbe@sha256:48515bbef9780ce32bdadd6e236d344217009fec4ed47a7809f41707ef8b8551",
        "OSFamily": "Linux",
        "Pool": "linux_x64_gpu",
    },
)

platform(
    name = "tf_linux_gpu",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
        "@bazel_tools//tools/cpp:clang",
    ],
    exec_properties = {
        "container-image": "docker://rocm/tensorflow-build@sha256:7cd444ac48657fee2f5087fbda7766266704d3f8fb2299f681952ae4eabed060",
        "OSFamily": "Linux",
        "Pool": "linux_x64_gpu",
    },
)
