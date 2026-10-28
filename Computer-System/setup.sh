#!/usr/bin/env bash
# Source this file in Bash or Zsh; execution cannot update the parent shell.
if [ -n "${BASH_VERSION-}" ]; then
    if [ "${BASH_SOURCE[0]}" = "$0" ]; then
        printf 'ERROR: source this script: source /path/to/Computer-System/setup.sh\n' >&2
        exit 1
    fi
elif [ -n "${ZSH_VERSION-}" ]; then
    case "${ZSH_EVAL_CONTEXT-}" in
        *:file) ;;
        *) printf 'ERROR: source this script: source /path/to/Computer-System/setup.sh\n' >&2
           exit 1 ;;
    esac
else
    printf 'ERROR: setup.sh supports Bash and Zsh.\n' >&2
    return 1 2>/dev/null || exit 1
fi

_cssys_setup() {
    local script_path project_root python_bin python_config python_libdir host_os
    local library_path library_search_path build_info for_build="${CSSYS_SETUP_FOR_BUILD:-0}"
    if [ -n "${BASH_VERSION-}" ]; then
        script_path="${BASH_SOURCE[0]}"
    else
        script_path="${(%):-%x}"
    fi
    project_root="$(cd -- "$(dirname -- "$script_path")" && pwd -P)" || return 1
    python_bin="$project_root/.venv/bin/python"
    if [ ! -x "$python_bin" ] || [ ! -r "$project_root/.venv/bin/activate" ]; then
        printf 'ERROR: missing virtual environment at %s/.venv. Run uv sync --locked from %s.\n' \
            "$project_root" "$project_root" >&2
        return 1
    fi

    # Validate the pin and development files before changing the caller's shell.
    if ! "$python_bin" - "$project_root/.python-version" <<'PY'
import ctypes
from pathlib import Path
import sys
import sysconfig

try:
    expected = Path(sys.argv[1]).read_text().strip()
    actual = ".".join(map(str, sys.version_info[:3]))
    if actual != expected:
        raise RuntimeError(f"Python {actual} does not match .python-version ({expected}); run uv sync --locked")
    header = Path(sysconfig.get_path("include")) / "Python.h"
    if not header.is_file():
        raise RuntimeError(f"missing Python header: {header}")
    library = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var("LDLIBRARY")
    ctypes.CDLL(str(library))
except Exception as exc:
    print(f"ERROR: Python environment check failed: {exc}", file=sys.stderr)
    sys.exit(1)
PY
    then
        return 1
    fi
    python_config="$("$python_bin" -c 'import sys, sysconfig; print(sysconfig.get_config_var("BINDIR") + "/python%d.%d-config" % sys.version_info[:2])')" || return 1
    python_libdir="$("$python_bin" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')" || return 1
    if [ ! -x "$python_config" ]; then
        printf 'ERROR: Python configuration is missing or not executable: %s\n' "$python_config" >&2
        return 1
    fi
    # uv's macOS python-config can omit -L for its own shared library.
    # Both Clang and GCC honor LIBRARY_PATH when linking embedded Python.
    library_search_path="${LIBRARY_PATH-}"
    case ":$library_search_path:" in
        *":$python_libdir:"*) ;;
        *) library_search_path="$python_libdir${library_search_path:+:$library_search_path}" ;;
    esac
    library_path="${LD_LIBRARY_PATH-}"
    host_os="$(uname -s)"
    if [ "$host_os" = Linux ]; then
        case ":$library_path:" in
            *":$python_libdir:"*) ;;
            *) library_path="$python_libdir${library_path:+:$library_path}" ;;
        esac
    fi

    if [ "$for_build" = 1 ]; then
        : # A rebuild must remain possible even if an older executable is broken.
    elif [ -e "$project_root/gem5/build/RISCV/gem5.opt" ]; then
        if [ ! -x "$project_root/gem5/build/RISCV/gem5.opt" ]; then
            printf 'ERROR: gem5.opt exists but is not executable.\n' >&2
            return 1
        fi
        if ! build_info="$(LD_LIBRARY_PATH="$library_path" "$project_root/gem5/build/RISCV/gem5.opt" --build-info 2>&1)"; then
            printf 'ERROR: gem5 --build-info failed:\n%s\n' "$build_info" >&2
            return 1
        fi
        case "$build_info" in
            *'USE_RISCV_ISA = True'*) ;;
            *) printf 'ERROR: gem5 does not report RISC-V support.\n' >&2
               return 1 ;;
        esac
    else
        printf 'WARNING: gem5 is not built yet. Run bash "%s/init.sh" when ready.\n' "$project_root" >&2
    fi

    . "$project_root/.venv/bin/activate" || return 1
    export PYTHON_CONFIG="$python_config"
    export LIBRARY_PATH="$library_search_path"
    if [ "$host_os" = Linux ]; then
        export LD_LIBRARY_PATH="$library_path"
    fi
    export GEM5_BIN="$project_root/gem5/build/RISCV/gem5.opt"
    printf 'CSSys environment loaded: Python %s\nGEM5_BIN=%s\n' \
        "$("$python_bin" --version 2>&1 | cut -d ' ' -f 2)" "$GEM5_BIN"
}

# Leave no helper function, directory change, or shell option change behind.
if _cssys_setup; then
    unset -f _cssys_setup
    return 0
else
    unset -f _cssys_setup
    return 1
fi
