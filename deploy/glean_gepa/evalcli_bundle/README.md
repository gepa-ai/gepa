# evalcli bundle

The public container builds without `evalcli` and can run `--fake_flow` for an
infrastructure smoke test. A real Cortex run needs a **Linux-built** evalcli
runtime at `/opt/evalcli/eval_cli`.

Do not copy a macOS Bazel output into this directory. The canonical production
image should be built from Scio, or from an internal Linux base image/artifact
that owns evalcli and its authentication mechanism.
