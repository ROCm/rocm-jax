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
#   1. Find the commit hash you want to pin to (e.g., from rocm-jaxlib-v0.8.2 branch)
#   2. Update XLA_COMMIT below

XLA_COMMIT = "9cc32ad7f36b1bf2c2f82ceb6efc264a6ecbf93e"

def repo():
    git_repository(
        name = "xla",
        remote = "https://github.com/ROCm/xla.git",
        commit = XLA_COMMIT,
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
