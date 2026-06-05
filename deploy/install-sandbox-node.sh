#!/usr/bin/env bash
# Provision a fresh Linux workstation as a hypotest sandbox node:
#   k3s (single-node)  +  gVisor (runsc)  +  agent-sandbox (controller + CRDs + sandbox-router)
# Mirrors the working aiapps-051525 setup, with the gotchas we hit baked in.
#
# RUN ON THE NEW WORKSTATION as a user with sudo (Linux, amd64 or arm64).
#
# Required env vars:
#   NODE_IP                 IP this node is reachable on (baked into the k3s API cert via --tls-san,
#                           so kubeconfig works from your Mac without insecure-skip-tls-verify)
#   AGENT_SANDBOX_VERSION   agent-sandbox release tag, e.g. v0.X.Y — match node 1 / the releases page
#                           (https://github.com/kubernetes-sigs/agent-sandbox/releases)
#
# Optional env var:
#   ROUTER_IMAGE            pre-built sandbox-router image to PULL (e.g. node 1's). If UNSET, the
#                           script BUILDS the router image on this node (needs docker) and imports it
#                           into k3s's containerd — no registry required (build per the upstream README).
#
# Examples:
#   # build the router locally on this node (no registry needed):
#   NODE_IP=10.110.41.99 AGENT_SANDBOX_VERSION=v0.3.0 ./deploy/install-sandbox-node.sh
#   # or reuse a pre-built/pushed image (e.g. node 1's):
#   NODE_IP=10.110.41.99 AGENT_SANDBOX_VERSION=v0.3.0 ROUTER_IMAGE=myreg/sandbox-router:latest \
#     ./deploy/install-sandbox-node.sh
set -euo pipefail

NODE_IP="${NODE_IP:?set NODE_IP (this nodes reachable IP, for the k3s API cert SAN)}"
AGENT_SANDBOX_VERSION="${AGENT_SANDBOX_VERSION:?set AGENT_SANDBOX_VERSION (e.g. v0.X.Y; see agent-sandbox releases)}"
ROUTER_IMAGE="${ROUTER_IMAGE:-}"   # optional — if unset, step 6 builds it locally + imports into k3s

ARCH="$(uname -m)"  # x86_64 | aarch64 — gVisor publishes both
K3S_TMPL="/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl"
ROUTER_BASE="https://raw.githubusercontent.com/kubernetes-sigs/agent-sandbox/main/clients/python/agentic-sandbox-client/sandbox-router"
KC() { sudo k3s kubectl "$@"; }   # k3s bundles kubectl + crictl

# `kubectl wait --all` errors "no matching resources found" if the node object
# doesn't exist YET (k3s API still starting). Poll until it registers, then wait Ready.
wait_node_ready() {
  echo "    waiting for k3s node to register..."
  for _ in $(seq 1 40); do
    if [ -n "$(KC get nodes --no-headers 2>/dev/null)" ]; then break; fi
    sleep 3
  done
  KC wait --for=condition=Ready node --all --timeout=180s
}

echo "==> 1/6  gVisor (runsc + containerd-shim-runsc-v1) -> /usr/local/bin"
tmp="$(mktemp -d)"; ( cd "$tmp"
  base="https://storage.googleapis.com/gvisor/releases/release/latest/${ARCH}"
  for f in runsc runsc.sha512 containerd-shim-runsc-v1 containerd-shim-runsc-v1.sha512; do
    curl -fsSLO "${base}/${f}"
  done
  sha512sum -c runsc.sha512 -c containerd-shim-runsc-v1.sha512
  chmod a+rx runsc containerd-shim-runsc-v1
  sudo mv runsc containerd-shim-runsc-v1 /usr/local/bin/ )
rm -rf "$tmp"
runsc --version | head -1

echo "==> 2/6  k3s (single-node server): --tls-san ${NODE_IP}, world-readable kubeconfig"
curl -sfL https://get.k3s.io | sudo sh -s - \
  --tls-san "${NODE_IP}" \
  --write-kubeconfig-mode 644
wait_node_ready

echo "==> 3/6  wire runsc into k3s's containerd (config version 3 path) + restart"
# k3s ships containerd 2.x => config version 3 => runtimes live under
# 'io.containerd.cri.v1.runtime' (NOT the legacy grpc.v1.cri). k3s honors a
# config.toml.tmpl that {{ template "base" . }} extends; drop-in dirs are not honored here.
sudo mkdir -p "$(dirname "$K3S_TMPL")"
sudo tee "$K3S_TMPL" >/dev/null <<'EOF'
{{ template "base" . }}

[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.runsc]
  runtime_type = "io.containerd.runsc.v1"
EOF
sudo systemctl restart k3s
wait_node_ready
# the live daemon must register runsc (this is the check that was the whole saga on node 1)
if sudo k3s crictl info | grep -q '"runsc"'; then
  echo "    runsc registered in the live containerd ✓"
else
  echo "    ERROR: runsc not registered — check 'sudo k3s crictl info' and that the shim is on PATH" >&2
  exit 1
fi

echo "==> 4/6  RuntimeClass gvisor -> runsc"
KC apply -f - <<'EOF'
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: gvisor
handler: runsc
EOF

echo "==> 5/6  agent-sandbox core + extensions (${AGENT_SANDBOX_VERSION})"
rel="https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${AGENT_SANDBOX_VERSION}"
KC apply -f "${rel}/manifest.yaml"
KC apply -f "${rel}/extensions.yaml"

echo "==> 6/6  sandbox-router"
local_build=0
if [ -z "$ROUTER_IMAGE" ]; then
  # No prebuilt image given -> build it here and import into k3s's containerd (no registry needed).
  # NOTE: docker and k3s run SEPARATE containerds (as on node 1), so build with docker then
  # `ctr import` into k3s's. The Dockerfile only COPYs requirements.txt + sandbox_router.py.
  command -v docker >/dev/null 2>&1 || {
    echo "ERROR: ROUTER_IMAGE unset and docker not installed — install docker, or pass a pullable ROUTER_IMAGE." >&2
    exit 1
  }
  echo "    building sandbox-router image (per the upstream router README) -> sandbox-router:local"
  bd="$(mktemp -d)"
  for f in Dockerfile requirements.txt sandbox_router.py; do curl -fsSL "$ROUTER_BASE/$f" -o "$bd/$f"; done
  docker build -t sandbox-router:local "$bd"
  rm -rf "$bd"
  echo "    importing image into k3s's containerd (k8s.io namespace)"
  docker save sandbox-router:local | sudo k3s ctr -n k8s.io images import -
  ROUTER_IMAGE="sandbox-router:local"
  local_build=1
fi
echo "    deploying router (image=${ROUTER_IMAGE}, unauthenticated test mode)"
# Substitute the image, and flip ALLOW_UNAUTHENTICATED_ROUTER to "true": the manifest ships
# "false", which makes the router refuse to start without a ROUTER_AUTH_TOKEN. For prod, set
# ROUTER_AUTH_TOKEN (via a Secret) and leave this "false" instead. The second sed targets only
# the value line immediately following the ALLOW_UNAUTHENTICATED_ROUTER name.
curl -fsSL "$ROUTER_BASE/sandbox_router.yaml" \
  | sed "s|\${ROUTER_IMAGE}|${ROUTER_IMAGE}|g" \
  | sed '/ALLOW_UNAUTHENTICATED_ROUTER/{n;s/"false"/"true"/;}' \
  | KC apply -f -
if [ "$local_build" = 1 ]; then
  # Locally-imported image (in no registry): make kubelet use it instead of trying to pull.
  KC patch deployment sandbox-router-deployment --type=json \
    -p='[{"op":"add","path":"/spec/template/spec/containers/0/imagePullPolicy","value":"IfNotPresent"}]' || true
fi

echo
echo "============================================================"
echo "DONE on ${NODE_IP}. Status:"
KC get runtimeclass gvisor 2>/dev/null || true
KC get pods -A | grep -Ei 'agent-sandbox|sandbox-router' || true
echo
echo "Next:"
echo "  1) Kubeconfig onto your Mac (mode 644, no sudo needed now):"
echo "       scp ${USER}@${NODE_IP}:/etc/rancher/k3s/k3s.yaml ~/.kube/$(hostname)-k3s.yaml"
echo "       sed -i '' 's#127.0.0.1:6443#${NODE_IP}:6443#' ~/.kube/$(hostname)-k3s.yaml   # macOS sed"
echo "  2) Deploy the kernel warm pool (fill in image + Secret first):"
echo "       kubectl apply -f deploy/sandbox-template.example.yaml"
echo "  3) Smoke the full path from the Mac:"
echo "       kubectl port-forward svc/sandbox-router-svc 18080:8080 &"
echo "       KUBECONFIG=~/.kube/$(hostname)-k3s.yaml uv run --extra k8s python scripts/ping_sandbox.py \\"
echo "         --api-url http://localhost:18080"
echo "============================================================"
