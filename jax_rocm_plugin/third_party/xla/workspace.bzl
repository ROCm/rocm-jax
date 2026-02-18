#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# buildifier: disable=module-docstring
load("//third_party:repo.bzl", "dynamic_git_repository")

# To update XLA to a new revision, update XLA_COMMIT to the new git commit hash.
# To temporarily override at build time without editing this file:
#   bazel build --repo_env=XLA_COMMIT_OVERRIDE=<commit> //...
#   bazel build --repo_env=XLA_COMMIT_OVERRIDE=<url>@<commit> //...

XLA_COMMIT = "24c5f10ae8fc24aefd20b43c501ade7f66fd0cfd"

def repo():
    dynamic_git_repository(
        name = "xla",
        remote = "https://github.com/ROCm/xla.git",
        commit = XLA_COMMIT,
        patch_file = [],
    )
