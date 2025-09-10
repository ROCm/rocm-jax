load("@rules_cc//cc:defs.bzl", "cc_toolchain", "cc_toolchain_suite")
#load(":cc_toolchain_config.bzl", "cc_toolchain_config")

package(default_visibility = ["//visibility:public"])

platform(
    name = "linux_x64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
        "@bazel_tools//tools/cpp:clang",
    ],
    exec_properties = {
	"container-image": "ghcr.io/rocm/jax-manylinux_2_28_x86_64_rocm700@sha256:4cd60135bd4b275c5c46e1453f87a3c9c023421007389f916da888183d439b53",
        "OSFamily": "Linux",
    },
)

