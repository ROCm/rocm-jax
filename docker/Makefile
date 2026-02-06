manylinux: manylinux_2_28_rocm7.2.0

dev: dev_ubuntu24.04_rocm7.2.0

manylinux_2_28_rocm7.2.0:
	docker build \
            -f manylinux/Dockerfile.jax-manylinux_2_28-rocm7.2.0 \
            -t jax-manylinux_2_28-rocm7.2.0:latest \
            .

dev_ubuntu24.04_rocm7.2.0:
	docker build \
            -f devsetup/Dockerfile.jax-dev-ubuntu24-rocm7.2.0 \
            -t jax-dev-ubuntu24-rocm7.2.0 \
	    .

