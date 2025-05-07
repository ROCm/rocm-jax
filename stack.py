
import os

JAX_REPL_URL = "https://github.com/rocm/jax"
XLA_REPL_URL = "https://github.com/rocm/xla"


XLA_DEV_BRANCH = "rocm-jaxlib-v0.5.0"


# dev mode setup

# clone jax
# clone xla
# setup build/install script for above
# setup test script for above


def setup_development():
    # clone jax repo for jax test case source code
    cmd = ["git", "clone"]
    cmd.append(JAX_REPL_URL)

    subprocess.check_run(cmd)

    # clone xla from source for building jax_rocm_plugin
    cmd = ["git", "clone"]
    cmd.extend(["--branch", XLA_DEV_BRANCH])
    cmd.append(XLA_REPL_URL)

    subprocess.check_run(cmd)

    # create build/install/test script



# build mode setup

# install jax/jaxlib from known versions
# setup build/install/test script

def setup_build():
    pass




def main():
    pass




if __name__ == "__main__":
    main()
