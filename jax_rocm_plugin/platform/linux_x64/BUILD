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
	#"container-image": "docker://rocm/tensorflow-build:2.19-noble-python3.12-rocm6.4.0@sha256:7f03657c78a71f63adcaa996bdfdbc1c276b55594076cc47365b8713b2030fdd",
	"container-image": "docker://938540721288.dkr.ecr.us-east-1.amazonaws.com/amd/rbe@sha256:d77247cae361ba785c70d5da712b3bb187526dd61bc4f1b2bdc6df53a1f66c2c",
        "OSFamily": "Linux",
    },
)

