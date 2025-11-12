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
        "container-image": "docker://938540721288.dkr.ecr.us-east-1.amazonaws.com/amd/jax-test@sha256:5ed826cb1e37e3a044f8b6f437368fe9104a64841c61ca10e28de8920b24a351",
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
