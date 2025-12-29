#!/usr/bin/env bash

#!/usr/bin/env bash
set -e

env
ls -alsh $PYTHON_RUNFILES

export PYTHONPATH=$TEST_SRCDIR/jax:$PYTHONPATH
export PYTHONPATH=$TEST_SRCDIR/jaxlib:$PYTHONPATH
export PYTHONPATH=$TEST_SRCDIR/jax/jax:$PYTHONPATH
export PYTHONPATH=$TEST_SRCDIR/jax/jax/_src:$PYTHONPATH
#export PYTHONPATH="$RUNFILES_DIR/jax:$RUNFILES_DIR/jax/jaxlib:$PYTHONPATH"
#export PYTHONPATH="$RUNFILES_DIR/pypi_numpy/site-packages:$PYTHONPATH"

exec "$@"
