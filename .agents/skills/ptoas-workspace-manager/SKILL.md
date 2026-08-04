---
name: ptoas-workspace-manager
description: Create, inspect, and safely retire isolated PTOAS feature workspaces. Use when developers need parallel PTOAS worktrees with per-worktree virtual environments and build directories while sharing an existing LLVM_BUILD_DIR, base Python/CANN installation, and ccache; also use when checking whether a merged feature workspace is safe to destroy.
---

# PTOAS workspace manager

Use the bundled `scripts/ptoas_workspace.py` instead of hand-writing the
worktree, venv, and editable-install commands. It creates one Git worktree,
one `.venv`, and one PTOAS CMake build directory per feature while reusing the
developer-selected LLVM build and base Python environment.

## Operating model

- Share a compatible LLVM/MLIR build (`LLVM_BUILD_DIR`) and optional `ccache`.
- Do not install PTOAS into a shared conda environment. Install it editable
  into the workspace's `.venv`.
- Create the venv from an explicitly selected `--base-python`. The default is
  the current interpreter, so the skill never assumes a conda environment
  name or path.
- CANN is not copied. Store only the optional environment-script path in the
  workspace metadata and generated `env.sh`; source it when Bisheng, simulator,
  or NPU execution is needed.
- Keep each workspace's `PTO_BUILD_DIR` separate. Sharing it across worktrees
  is unsafe because CMake caches source paths and generated files.

## Create a workspace

From any checkout of PTOAS, choose a workspace root and the existing LLVM
build. The base Python can be conda, a system Python, or another compatible
interpreter:

```bash
python3 .agents/skills/ptoas-workspace-manager/scripts/ptoas_workspace.py create \
  --repo /path/to/PTOAS \
  --name feature-a \
  --workspace-root /data/ptoas-dev/workspaces \
  --build-root /data/ptoas-dev/builds \
  --llvm-build-dir /data/ptoas-dev/llvm-project/build-vpto21-py311 \
  --base-python /opt/conda/envs/pto/bin/python \
  --ccache-dir /data/ptoas-dev/.ccache \
  --cann-env /usr/local/Ascend/cann/set_env.sh
```

The command creates branch `codex/feature-a` by default, unless `--branch`
is supplied. It writes `.ptoas-workspace.json` and `env.sh` inside the
workspace, creates a venv with `--system-site-packages`, and runs the repo's
`quick_install.sh` with an isolated `PTO_BUILD_DIR`. Use `--skip-install` when
only the worktree and venv should be prepared.

After creation:

```bash
source /data/ptoas-dev/workspaces/feature-a/env.sh
ptoas --version
```

If the selected base Python cannot create venvs, install its venv support or
use another interpreter. If the workspace needs Python dependencies that must
not be shared through system site packages, add `--no-system-site-packages`.

For repeated use, `PTOAS_REPO`, `PTOAS_WORKSPACE_ROOT`, `PTOAS_BUILD_ROOT`,
`LLVM_BUILD_DIR`, `PTOAS_BASE_PYTHON`, `CANN_ENV_SCRIPT`, and `CCACHE_DIR` can
provide machine-specific defaults. Command-line options take precedence.

## Inspect and retire a workspace

Always inspect before removing it:

```bash
python3 .agents/skills/ptoas-workspace-manager/scripts/ptoas_workspace.py status \
  --workspace /data/ptoas-dev/workspaces/feature-a
```

Retirement is eligible only when all of these conditions hold (the manager's
own generated `.venv/`, metadata, and `env.sh` are excluded from this
source-change check and removed as part of destruction):

1. Git reports no staged, unstaged, or untracked changes in the worktree.
2. A GitHub pull request for the workspace branch exists and is merged.
3. The worktree's HEAD is an ancestor of the merged PR's head commit, so no
   local commit — pushed or not — is missing from the merged PR.

The script uses `gh` for the PR check. It fails closed when `gh` is absent,
authentication is unavailable, the repository cannot be identified, no
merged PR is found, the PR head commit cannot be fetched, or any local
commit is not contained in it. A closed-but-unmerged PR is not sufficient.

Check eligibility without deleting anything:

```bash
python3 .agents/skills/ptoas-workspace-manager/scripts/ptoas_workspace.py destroy \
  --workspace /data/ptoas-dev/workspaces/feature-a
```

Only after reviewing the report, perform the destructive operation explicitly:

```bash
python3 .agents/skills/ptoas-workspace-manager/scripts/ptoas_workspace.py destroy \
  --workspace /data/ptoas-dev/workspaces/feature-a --yes
```

The command removes the Git worktree and its per-worktree build directory. It
does not remove the shared LLVM build, CANN installation, ccache, or the local
branch unless `--delete-branch` is also supplied. It refuses to remove the
current working directory or a path that is not a registered Git worktree.

## Agent procedure

When the user asks to create a workspace, first resolve the repository path,
feature name, base ref, workspace/build roots, `LLVM_BUILD_DIR`, and base
Python from local context or explicit arguments. Do not silently select a
different LLVM build or Python ABI. Run `create`, then report the generated
workspace path and `env.sh`.

When the user asks to clean up or destroy one, run `status` or a dry-run
`destroy` first. Never bypass either gate and never recursively delete a
workspace manually. If a gate fails, report the exact reason and leave all
files untouched.

For implementation details and option behavior, read
`references/workspace-layout.md` and inspect the bundled script only when
needed.
