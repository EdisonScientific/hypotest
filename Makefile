.PHONY: image server help

help:
	@echo "Available targets:"
	@echo "  make server CONFIG=<path>  - Launch the dataset server with the given config file"
	@echo "  make image                 - Build the Docker image for interpreter-env"
	@echo "  make help                  - Show this help message"

server:
	@test -n "$(CONFIG)" || (echo "Error: CONFIG is required. Usage: make server CONFIG=path/to/config.yaml" && exit 1)
	uv run -p /app/kernel_env/bin/python src/hypotest/dataset_server.py $(CONFIG)

image:
	DOCKER_BUILDKIT=1 docker build -t interpreter-env:latest .
