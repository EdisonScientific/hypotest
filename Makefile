.PHONY: image

image:
	DOCKER_BUILDKIT=1 docker build -t interpreter-env:latest .
