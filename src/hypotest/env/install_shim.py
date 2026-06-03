# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Install-command shim: intercept pip / conda / apt-get / R install.packages /
BiocManager::install at the kernel-container boundary. Already-installed
packages short-circuit with a "[pre-installed]" message; the real installer
only runs when a package is genuinely missing or when the model passes a
native force flag. See features/infra_fault_tolerance/infra_mitigations_handoff.md §3.4.

Pure data + one I/O helper, no dependencies on the kernel / nbformat / ray
stack. Import from here rather than from interpreter_env.py when unit-testing.
"""

from __future__ import annotations

from pathlib import Path

_PIP_SHIM_BASH = r"""#!/usr/bin/env bash
# Install shim for pip.
# See features/infra_fault_tolerance/infra_mitigations_handoff.md §3.4.
# - Non-install subcommands pass through.
# - --force-reinstall / --upgrade / -U and -r / -e / URL / path / --find-links / --index-url
#   all bypass the short-circuit and hit the real installer (with a 120s timeout).
# - Otherwise, for each positional package spec, check if already installed; skip if so.
# - Version pins without --force-reinstall produce a "[version-mismatch]" warning and skip.
set -euo pipefail

REAL_PIP="/app/kernel_env/bin/pip"
LOG="${INSTALL_SHIM_LOG:-/dev/null}"
NOW_MS() { date +%s%3N; }

if [ "${1:-}" != "install" ]; then
    exec "$REAL_PIP" "$@"
fi
shift

force=0
passthrough=0
args=()
prev_is_r=0
for arg in "$@"; do
    if [ "$prev_is_r" = 1 ]; then
        passthrough=1
        args+=("$arg")
        prev_is_r=0
        continue
    fi
    case "$arg" in
        --force-reinstall|--upgrade|-U)
            force=1
            args+=("$arg")
            ;;
        -r|--requirement)
            passthrough=1
            prev_is_r=1
            args+=("$arg")
            ;;
        -e|--editable|--find-links|--index-url|--extra-index-url)
            passthrough=1
            args+=("$arg")
            ;;
        git+*|http*|*.whl|*.tar.gz|/*|./*)
            passthrough=1
            args+=("$arg")
            ;;
        *)
            args+=("$arg")
            ;;
    esac
done

if [ "$force" = 1 ] || [ "$passthrough" = 1 ]; then
    t0=$(NOW_MS)
    set +e
    timeout 120 "$REAL_PIP" install "${args[@]}"
    rc=$?
    set -e
    t1=$(NOW_MS)
    outcome="passthrough"
    [ "$force" = 1 ] && outcome="passthrough_force"
    printf '{"tool":"pip","outcome":"%s","elapsed_ms":%d,"rc":%d}\n' \
        "$outcome" $((t1 - t0)) "$rc" >> "$LOG" 2>/dev/null || true
    exit "$rc"
fi

to_install=()
for spec in "${args[@]}"; do
    case "$spec" in -*) continue ;; esac
    pkg="${spec%%[=<>!]*}"
    pinned=""
    if [ "$pkg" != "$spec" ]; then
        pinned="${spec#"$pkg"}"
    fi
    t0=$(NOW_MS)
    if "$REAL_PIP" show "$pkg" >/dev/null 2>&1; then
        installed_ver="$("$REAL_PIP" show "$pkg" 2>/dev/null | awk '/^Version:/ {print $2}')"
        t1=$(NOW_MS)
        if [ -n "$pinned" ]; then
            echo "[pre-installed][version-mismatch] $pkg pinned to $pinned but $installed_ver is installed; using installed version. Pass --force-reinstall to override."
            printf '{"tool":"pip","pkg":"%s","pin":"%s","installed":"%s","outcome":"skipped_version_mismatch","elapsed_ms":%d}\n' \
                "$pkg" "$pinned" "$installed_ver" $((t1 - t0)) >> "$LOG" 2>/dev/null || true
        else
            echo "[pre-installed] $pkg ($installed_ver) already available, skipping"
            printf '{"tool":"pip","pkg":"%s","installed":"%s","outcome":"skipped","elapsed_ms":%d}\n' \
                "$pkg" "$installed_ver" $((t1 - t0)) >> "$LOG" 2>/dev/null || true
        fi
    else
        to_install+=("$spec")
    fi
done

if [ "${#to_install[@]}" -gt 0 ]; then
    t0=$(NOW_MS)
    set +e
    timeout 120 "$REAL_PIP" install "${to_install[@]}"
    rc=$?
    set -e
    t1=$(NOW_MS)
    if [ "$rc" = 0 ]; then
        outcome=installed
    elif [ "$rc" = 124 ]; then
        outcome=install_timeout
    else
        outcome=install_failed
    fi
    printf '{"tool":"pip","pkgs":"%s","outcome":"%s","elapsed_ms":%d,"rc":%d}\n' \
        "${to_install[*]}" "$outcome" $((t1 - t0)) "$rc" >> "$LOG" 2>/dev/null || true
    exit "$rc"
fi
exit 0
"""


_CONDA_SHIM_BASH = r"""#!/usr/bin/env bash
# Install shim for conda. Same shape as pip shim; escape hatch is --force-reinstall.
set -euo pipefail

REAL_CONDA="/app/kernel_env/bin/conda"
if [ ! -x "$REAL_CONDA" ]; then
    REAL_CONDA="$(command -v conda || echo conda)"
fi
LOG="${INSTALL_SHIM_LOG:-/dev/null}"
NOW_MS() { date +%s%3N; }

if [ "${1:-}" != "install" ]; then
    exec "$REAL_CONDA" "$@"
fi
shift

force=0
args=()
for arg in "$@"; do
    case "$arg" in
        --force-reinstall|--force|--update-deps) force=1; args+=("$arg");;
        *) args+=("$arg");;
    esac
done

if [ "$force" = 1 ]; then
    t0=$(NOW_MS)
    set +e
    timeout 120 "$REAL_CONDA" install "${args[@]}"
    rc=$?
    set -e
    t1=$(NOW_MS)
    printf '{"tool":"conda","outcome":"passthrough_force","elapsed_ms":%d,"rc":%d}\n' \
        $((t1 - t0)) "$rc" >> "$LOG" 2>/dev/null || true
    exit "$rc"
fi

to_install=()
for spec in "${args[@]}"; do
    case "$spec" in -*) continue ;; esac
    pkg="${spec%%[=<>!]*}"
    t0=$(NOW_MS)
    if "$REAL_CONDA" list -f "$pkg" 2>/dev/null | awk '!/^#/ && NF' | grep -q "^$pkg "; then
        installed_ver="$("$REAL_CONDA" list -f "$pkg" 2>/dev/null | awk '!/^#/ && NF {print $2; exit}')"
        t1=$(NOW_MS)
        echo "[pre-installed] $pkg ($installed_ver) already available via conda, skipping"
        printf '{"tool":"conda","pkg":"%s","installed":"%s","outcome":"skipped","elapsed_ms":%d}\n' \
            "$pkg" "$installed_ver" $((t1 - t0)) >> "$LOG" 2>/dev/null || true
    else
        to_install+=("$spec")
    fi
done

if [ "${#to_install[@]}" -gt 0 ]; then
    t0=$(NOW_MS)
    set +e
    timeout 120 "$REAL_CONDA" install -y "${to_install[@]}"
    rc=$?
    set -e
    t1=$(NOW_MS)
    if [ "$rc" = 0 ]; then outcome=installed
    elif [ "$rc" = 124 ]; then outcome=install_timeout
    else outcome=install_failed; fi
    printf '{"tool":"conda","pkgs":"%s","outcome":"%s","elapsed_ms":%d,"rc":%d}\n' \
        "${to_install[*]}" "$outcome" $((t1 - t0)) "$rc" >> "$LOG" 2>/dev/null || true
    exit "$rc"
fi
exit 0
"""


_APT_SHIM_BASH = r"""#!/usr/bin/env bash
# Install shim for apt-get. Escape hatch: --reinstall.
set -euo pipefail

REAL_APT="/usr/bin/apt-get"
LOG="${INSTALL_SHIM_LOG:-/dev/null}"
NOW_MS() { date +%s%3N; }

if [ "${1:-}" != "install" ]; then
    exec "$REAL_APT" "$@"
fi
shift

force=0
args=()
for arg in "$@"; do
    case "$arg" in
        --reinstall) force=1; args+=("$arg");;
        *) args+=("$arg");;
    esac
done

if [ "$force" = 1 ]; then
    t0=$(NOW_MS)
    set +e
    timeout 120 "$REAL_APT" install -y "${args[@]}"
    rc=$?
    set -e
    t1=$(NOW_MS)
    printf '{"tool":"apt-get","outcome":"passthrough_force","elapsed_ms":%d,"rc":%d}\n' \
        $((t1 - t0)) "$rc" >> "$LOG" 2>/dev/null || true
    exit "$rc"
fi

to_install=()
for spec in "${args[@]}"; do
    case "$spec" in -*) continue ;; esac
    pkg="${spec%%=*}"
    t0=$(NOW_MS)
    if dpkg -s "$pkg" >/dev/null 2>&1; then
        installed_ver="$(dpkg -s "$pkg" 2>/dev/null | awk '/^Version:/ {print $2}')"
        t1=$(NOW_MS)
        echo "[pre-installed] $pkg ($installed_ver) already available via dpkg, skipping"
        printf '{"tool":"apt-get","pkg":"%s","installed":"%s","outcome":"skipped","elapsed_ms":%d}\n' \
            "$pkg" "$installed_ver" $((t1 - t0)) >> "$LOG" 2>/dev/null || true
    else
        to_install+=("$spec")
    fi
done

if [ "${#to_install[@]}" -gt 0 ]; then
    t0=$(NOW_MS)
    set +e
    timeout 120 "$REAL_APT" install -y "${to_install[@]}"
    rc=$?
    set -e
    t1=$(NOW_MS)
    if [ "$rc" = 0 ]; then outcome=installed
    elif [ "$rc" = 124 ]; then outcome=install_timeout
    else outcome=install_failed; fi
    printf '{"tool":"apt-get","pkgs":"%s","outcome":"%s","elapsed_ms":%d,"rc":%d}\n' \
        "${to_install[*]}" "$outcome" $((t1 - t0)) "$rc" >> "$LOG" 2>/dev/null || true
    exit "$rc"
fi
exit 0
"""


_R_SHIM_CODE = r"""
# ---- install-command shim (pre-installed short-circuit; use force=TRUE to override) ----
# See features/infra_fault_tolerance/infra_mitigations_handoff.md §3.4.
local({
  log_path <- Sys.getenv("INSTALL_SHIM_LOG")
  shim_log <- function(entry) {
    if (nzchar(log_path)) {
      tryCatch({
        line <- jsonlite::toJSON(entry, auto_unbox = TRUE, null = "null")
        cat(as.character(line), "\n", file = log_path, append = TRUE, sep = "")
      }, error = function(e) NULL)
    }
  }

  has_pkg <- function(p) requireNamespace(p, quietly = TRUE)

  shim_install_factory <- function(tool_name, real_fn) {
    function(pkgs = character(), ...) {
      dots <- list(...)
      force <- isTRUE(dots$force) || isTRUE(dots$update)
      if (is.null(pkgs) || length(pkgs) == 0) return(invisible(NULL))
      t0 <- proc.time()[["elapsed"]]
      if (force) {
        setTimeLimit(elapsed = 120, transient = TRUE)
        on.exit(setTimeLimit(elapsed = Inf, transient = TRUE), add = TRUE)
        tryCatch(real_fn(pkgs, ...), error = function(e) message(conditionMessage(e)))
        shim_log(list(tool = tool_name, pkgs = pkgs, outcome = "passthrough_force",
                      elapsed_ms = as.integer(1000 * (proc.time()[["elapsed"]] - t0))))
        return(invisible(NULL))
      }
      is_present <- vapply(pkgs, has_pkg, logical(1))
      already <- pkgs[is_present]; missing <- pkgs[!is_present]
      if (length(already)) {
        versions <- vapply(already, function(p) tryCatch(as.character(utils::packageVersion(p)),
                                                         error = function(e) "?"),
                           character(1))
        message(sprintf("[pre-installed] %s already available, skipping",
                        paste(sprintf("%s (%s)", already, versions), collapse = ", ")))
        shim_log(list(tool = tool_name, pkgs = already, outcome = "skipped",
                      elapsed_ms = as.integer(1000 * (proc.time()[["elapsed"]] - t0))))
      }
      if (length(missing)) {
        setTimeLimit(elapsed = 120, transient = TRUE)
        on.exit(setTimeLimit(elapsed = Inf, transient = TRUE), add = TRUE)
        tryCatch(real_fn(missing, ...), error = function(e) message(conditionMessage(e)))
        shim_log(list(tool = tool_name, pkgs = missing, outcome = "installed",
                      elapsed_ms = as.integer(1000 * (proc.time()[["elapsed"]] - t0))))
      }
      invisible(NULL)
    }
  }

  if (requireNamespace("BiocManager", quietly = TRUE)) {
    real_bm <- utils::getFromNamespace("install", "BiocManager")
    tryCatch(
      utils::assignInNamespace("install", shim_install_factory("BiocManager", real_bm), ns = "BiocManager"),
      error = function(e) message("[install-shim] could not override BiocManager::install: ", conditionMessage(e))
    )
  }
  # Shadow install.packages in the user's global environment. Escape hatch:
  # call utils::install.packages(...) explicitly to bypass the shim.
  real_ip <- utils::install.packages
  assign("install.packages", shim_install_factory("install.packages", real_ip),
         envir = globalenv())
})
"""


# R user-profile header: make R_LIBS_USER the first lib path. The install shim
# (_R_SHIM_CODE) is appended after it. Kept here so the enroot bash and the k8s
# kernel server write byte-identical Rprofiles.
_RPROFILE_HEADER = (
    '.local_lib <- Sys.getenv("R_LIBS_USER")\nif (nzchar(.local_lib)) .libPaths(unique(c(.local_lib, .libPaths())))\n'
)


def write_workspace_config(work_dir: Path, runtime_path: str, index_url: str | None = None) -> None:
    """Write the agent workspace install scaffolding into ``work_dir``.

    Creates ``pydeps/``, ``pip-cache/``, ``r_libs/``; a ``pip.conf`` whose
    ``target``/``cache-dir`` point at ``runtime_path`` (where the workspace lives
    when the kernel runs — the same dir for k8s, the in-container mount for
    enroot) and which carries ``index_url`` when given (the runtime supply-chain
    cutoff proxy; pip.conf overrides /etc/pip.conf, so it must re-state it); the
    pip/conda/apt + R install shims; and the ``Rprofile`` (libPaths + R shim).

    Single source of truth for the install model so the enroot and k8s kernel
    paths can't drift. Pass ``index_url=None`` for enroot (no in-pod proxy).
    """
    wd = Path(work_dir)
    for sub in ("pydeps", "pip-cache", "r_libs"):
        (wd / sub).mkdir(parents=True, exist_ok=True)
    pip_conf = [
        "[global]",
        "disable-pip-version-check = true",
        "no-input = true",
        f"cache-dir = {runtime_path}/pip-cache",
        f"target = {runtime_path}/pydeps",
    ]
    if index_url:
        pip_conf.append(f"index-url = {index_url}")
    (wd / "pip.conf").write_text("\n".join(pip_conf) + "\n")
    _write_install_shims(wd)
    (wd / "Rprofile").write_text(_RPROFILE_HEADER + _R_SHIM_CODE)


def workspace_env(runtime_path: str) -> dict[str, str]:
    """Env vars the kernel needs for the workspace install model.

    ``PYTHONPATH`` and ``PATH`` are the SEGMENTS to prepend to any existing
    value; the rest are absolute. Used by both the enroot bash (to generate its
    ``export`` lines) and the k8s kernel server (to update ``os.environ``).
    """
    return {
        "PYTHONPATH": f"{runtime_path}/pydeps",
        "PATH": f"{runtime_path}/.install_shim/bin",
        "PIP_CONFIG_FILE": f"{runtime_path}/pip.conf",
        "INSTALL_SHIM_LOG": f"{runtime_path}/.install_shim/log",
        "R_LIBS_USER": f"{runtime_path}/r_libs",
        "R_PROFILE_USER": f"{runtime_path}/Rprofile",
    }


def bash_export_block(runtime_path: str) -> str:
    """Render ``workspace_env`` as ``export VAR=...`` lines for the enroot bash.

    ``runtime_path`` may be a bash variable like ``$WORKDIR`` (kept literal so the
    shell expands it). ``PYTHONPATH``/``PATH`` are prepended to any existing value.
    Same single source as the k8s server's ``os.environ`` update, so they can't drift.
    """
    lines = []
    for key, seg in workspace_env(runtime_path).items():
        if key in {"PYTHONPATH", "PATH"}:
            lines.append(f'export {key}="{seg}:${{{key}}}"')
        else:
            lines.append(f'export {key}="{seg}"')
    return "\n".join(lines)


def _write_install_shims(work_dir: Path) -> None:
    """Write pip/conda/apt-get wrappers + R shim into the workspace dir.

    Everything goes under `$WORKDIR/.install_shim/` (dot-prefixed so it's
    hidden from `list_dir` by default and doesn't clutter the file listing
    the agent sees). The kernel-init bash script prepends
    `$WORKDIR/.install_shim/bin` to `PATH` so these wrappers intercept tool
    calls; the R shim is appended to the generated Rprofile; the per-rollout
    invocation log lands at `$WORKDIR/.install_shim/log`. See
    features/infra_fault_tolerance/infra_mitigations_handoff.md.
    """
    shim_dir = work_dir / ".install_shim"
    bin_dir = shim_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name, content in (
        ("pip", _PIP_SHIM_BASH),
        ("conda", _CONDA_SHIM_BASH),
        ("apt-get", _APT_SHIM_BASH),
    ):
        wrapper_path = bin_dir / name
        wrapper_path.write_text(content)
        wrapper_path.chmod(0o755)
    (shim_dir / "r_shim.R").write_text(_R_SHIM_CODE)
