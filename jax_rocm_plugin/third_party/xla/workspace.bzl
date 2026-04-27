#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# buildifier: disable=module-docstring
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# To update XLA:
#   1. Find the commit hash you want to pin to (e.g., from rocm-jaxlib-v0.9.2 branch)
#   2. Update XLA_COMMIT below

XLA_COMMIT = "60f77b59e233ffb97d039086f0531ab75d7e0181"

def repo():
    git_repository(
        name = "xla",
        remote = "https://github.com/ROCm/xla.git",
        commit = XLA_COMMIT,
        patches = [
            # Fix for zstd assembly compilation with LLVM-18's cet.h header
            # The cet.h include path was not in cxx_builtin_include_directories
            "//third_party/xla:0001-Add-clang-resource-dir-include-path.patch",
            # Upgrade rules_python 1.8.4 -> 1.8.5 to fix %interpreter_args% SyntaxError
            "//third_party/xla:0002-upgrade-rules-python-to-1.8.5.patch",
        ],
        patch_args = ["-p1"],
    )

    # For development, one often wants to make changes to the XLA repository as well
    # as the JAX repository. You can override the pinned repository above with a
    # local checkout by either:
    # a) overriding the XLA repository on the build.py command line by passing a flag
    #    like:
    #    python build/build.py build --local_xla_path=/path/to/xla
    #    or
    # b) by commenting out the git_repository above and uncommenting the following:
    # local_repository(
    #    name = "xla",
    #    path = "/path/to/xla",
    # )
