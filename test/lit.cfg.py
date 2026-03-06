import os
import lit.formats


config.name = "PTOAS"
config.test_format = lit.formats.ShTest(execute_external=True)

# Keep discovery focused on lit-style tests.
config.suffixes = [".mlir"]
config.excludes = [
    "CMakeLists.txt",
    "README.md",
    "lit.cfg.py",
    "oplib",
    "oplib_ccec",
]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.test_source_root


def _resolve_ptoas_bin():
    env_bin = os.environ.get("PTOAS_BIN")
    if env_bin:
        return env_bin

    repo_root = os.path.abspath(os.path.join(config.test_source_root, ".."))
    candidate = os.path.join(repo_root, "build", "tools", "ptoas", "ptoas")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate

    return "ptoas"


def _prepend_path(path_var, entry):
    if not entry:
        return path_var
    if not path_var:
        return entry
    return entry + os.pathsep + path_var


ptoas_bin = _resolve_ptoas_bin()
ptoas_dir = os.path.dirname(ptoas_bin) if os.path.isabs(ptoas_bin) else ""

path_env = config.environment.get("PATH", os.environ.get("PATH", ""))
if ptoas_dir:
    path_env = _prepend_path(path_env, ptoas_dir)
config.environment["PATH"] = path_env

# Keep RUN lines using bare `ptoas` stable regardless of shell cwd.
if os.path.isabs(ptoas_bin):
    config.substitutions.append(("ptoas", ptoas_bin))
