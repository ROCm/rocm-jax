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
	"container-image": "docker://938540721288.dkr.ecr.us-east-1.amazonaws.com/amd/rbe@sha256:ff07b8fa3425b1317bd19b8f9f63dd7e70b01769def1536d9d7076ce0fd60760",
        "OSFamily": "Linux",
    },
)

