#!/bin/sh
# Entrypoint for the bundled hypotest dataset-server image.
#
# Starts the package-date-cutoff proxy (conda repodata + pip Simple) — the
# runtime half of the supply-chain cutoff. Neither conda nor real pip has a
# native date option or a dated-index service, so this local proxy enforces the
# cutoff; the baked /app/miniconda/.condarc and /etc/pip.conf route conda and pip
# through it. (apt and R use dated snapshot services via pure config.) The proxy
# binds 127.0.0.1 (pod-local) and idles until the first install triggers a fetch.
# If it dies, conda/pip installs fail closed (no unbounded install) while the
# server keeps running.
#
# Then hand off to the dataset server (the image CMD, or an overriding command).
set -e

python /opt/cutoff_proxy.py --port 8723 --cutoff "${CUTOFF_DATE:-$(date -u +%Y-%m-%d)}" &

exec "$@"
