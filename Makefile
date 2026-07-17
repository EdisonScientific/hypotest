.PHONY: image image-core kernel-image kernel-image-core server-image server-image-core server help

# Supply-chain cutoff: no package published after this date is installed.
# Measured at build time; override to pin a known-good date, e.g.
#   make image BUILD_CUTOFF_DATE=2026-05-01
BUILD_CUTOFF_DATE ?= $(shell date -u +%Y-%m-%d)

# Platform for the amd64-only production (full) images. Overridable; on an
# arm64 host these build under emulation. The core images build natively.
PLATFORM ?= linux/amd64

KERNEL_IMAGE ?= interpreter-env
SERVER_IMAGE ?= hypotest-server
KERNEL_SERVER_IMAGE ?= hypotest-kernel
# Optional, non-secret defaults for init-time capsule delivery. Per-allocation
# OpenSandbox env overrides both; leave empty for a fully generic image.
CAPSULE_SOURCE ?=
CAPSULE_KEY ?=

export DOCKER_BUILDKIT = 1

help:
	@echo "Build-date cutoff: BUILD_CUTOFF_DATE=$(BUILD_CUTOFF_DATE)"
	@echo ""
	@echo "Images:"
	@echo "  make image                 - Full kernel/exec image ($(KERNEL_IMAGE):latest, amd64). Backwards-compatible."
	@echo "  make image-core            - Lightweight kernel base ($(KERNEL_IMAGE):core, native arch) for local testing."
	@echo "  make server-image          - Bundled dataset-server image ($(SERVER_IMAGE):latest, amd64) on the full base."
	@echo "  make server-image-core     - Bundled dataset-server image ($(SERVER_IMAGE):core, native arch) on the core base."
	@echo "  make kernel-image          - Lean capsule-loading kernel server ($(KERNEL_SERVER_IMAGE):latest, amd64) for sandbox pods."
	@echo "  make kernel-image-core     - Lean kernel server ($(KERNEL_SERVER_IMAGE):core, native arch) for local testing."
	@echo ""
	@echo "Run:"
	@echo "  make server CONFIG=<path>  - Launch the dataset server on the host (uv)."
	@echo "  make help                  - Show this help message."

# Full kernel/exec image (amd64, production). Backwards-compatible default tag.
image:
	docker build \
		--platform $(PLATFORM) \
		--target full \
		--build-arg BUILD_CUTOFF_DATE=$(BUILD_CUTOFF_DATE) \
		-t $(KERNEL_IMAGE):latest -t $(KERNEL_IMAGE):full .

# Lightweight kernel base, native architecture (arm64 on Apple Silicon).
image-core:
	docker build \
		--target core \
		--build-arg BUILD_CUTOFF_DATE=$(BUILD_CUTOFF_DATE) \
		-t $(KERNEL_IMAGE):core .

# Bundled dataset-server image (amd64, production) on the full kernel base.
server-image: image
	docker build \
		--platform $(PLATFORM) \
		-f Dockerfile.server \
		--build-arg BASE_IMAGE=$(KERNEL_IMAGE):full \
		--build-arg BUILD_CUTOFF_DATE=$(BUILD_CUTOFF_DATE) \
		-t $(SERVER_IMAGE):latest .

# Bundled dataset-server image on the lightweight core base (local arm64 / Mac).
server-image-core: image-core
	docker build \
		-f Dockerfile.server \
		--build-arg BASE_IMAGE=$(KERNEL_IMAGE):core \
		--build-arg BUILD_CUTOFF_DATE=$(BUILD_CUTOFF_DATE) \
		-t $(SERVER_IMAGE):core .

# Lean kernel-server image (amd64, production) on the full kernel base — the image
# the sandbox infra runs (agent-sandbox pods / enroot). hypotest is installed
# --no-deps (no dataset-server/agent stack); see Dockerfile.kernel.
kernel-image: image
	docker build \
		--platform $(PLATFORM) \
		-f Dockerfile.kernel \
		--build-arg BASE_IMAGE=$(KERNEL_IMAGE):full \
		--build-arg BUILD_CUTOFF_DATE=$(BUILD_CUTOFF_DATE) \
		--build-arg CAPSULE_SOURCE=$(CAPSULE_SOURCE) \
		--build-arg CAPSULE_KEY=$(CAPSULE_KEY) \
		-t $(KERNEL_SERVER_IMAGE):latest .

# Lean kernel-server image on the lightweight core base (local arm64 / Mac).
kernel-image-core: image-core
	docker build \
		-f Dockerfile.kernel \
		--build-arg BASE_IMAGE=$(KERNEL_IMAGE):core \
		--build-arg BUILD_CUTOFF_DATE=$(BUILD_CUTOFF_DATE) \
		--build-arg CAPSULE_SOURCE=$(CAPSULE_SOURCE) \
		--build-arg CAPSULE_KEY=$(CAPSULE_KEY) \
		-t $(KERNEL_SERVER_IMAGE):core .

server:
	@test -n "$(CONFIG)" || (echo "Error: CONFIG is required. Usage: make server CONFIG=path/to/config.yaml" && exit 1)
	uv run python src/hypotest/dataset_server.py $(CONFIG)
