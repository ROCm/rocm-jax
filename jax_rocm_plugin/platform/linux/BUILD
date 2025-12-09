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
        "container-image": "docker://ghcr.io/rocm/jax-manylinux_2_28_x86_64_rocm702@sha256:3b7c7d786b72aa2524fbede28df822609106576d10deed6de7325164ebb724b7",
        "OSFamily": "Linux",
    },
)

platform(
    name = "ubuntu_gpu",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
        "@bazel_tools//tools/cpp:clang",
    ],
    exec_properties = {
        "container-image": "docker://ghcr.io/rocm/jax-ubu22.rocm710@sha256:0153279d93b796c9b9adc4d196e921388f9ee4ea31f2933e8310667ee880eecc",
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
