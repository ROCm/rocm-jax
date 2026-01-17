#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# buildifier: disable=module-docstring
load("//third_party:repo.bzl", "amd_http_archive")

# To update XLA to a new revision,
# a) update XLA_COMMIT to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/openxla/xla/archive/<git hash>.tar.gz | sha256sum
#    and update XLA_SHA256 with the result.

XLA_COMMIT = "33bdc4f6ee79ae3fa1e10e4afd6d229600fc8dc6"
XLA_SHA256 = "a4117a8801dce14434ecb918bba458322a0c53ef342284b5d9f2e5f112b252d8"

def repo():
    amd_http_archive(
        name = "xla",
        sha256 = XLA_SHA256,
        strip_prefix = "xla-{commit}".format(commit = XLA_COMMIT),
        urls = ["https://github.com/ROCm/xla/archive/{commit}.tar.gz".format(commit = XLA_COMMIT)],
        patch_file = [],
    )

    # For development, one often wants to make changes to the TF repository as well
    # as the JAX repository. You can override the pinned repository above with a
    # local checkout by either:
    # a) overriding the TF repository on the build.py command line by passing a flag
    #    like:
    #    python build/build.py build --local_xla_path=/path/to/xla
    #    or
    # b) by commenting out the http_archive above and uncommenting the following:
    # local_repository(
    #    name = "xla",
    #    path = "/path/to/xla",
    # )
