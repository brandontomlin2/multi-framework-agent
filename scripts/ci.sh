#!/usr/bin/env bash
# The CI check, runnable anywhere: GitHub Actions runs this same script.
# Builds a fresh virtualenv, installs the package with all extras, runs the offline tests.
#
#   scripts/ci.sh                 # run checks
#   scripts/ci.sh --post-status   # also post the result to the current commit on GitHub
#                                 # (needs the gh CLI, authenticated, with repo:status)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
VENV="$(mktemp -d)/venv"
"$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q -e ".[all]"

set +e
"$VENV/bin/python" -m pytest -q 2>&1 | tee /tmp/mfa-ci-pytest.log
status=${PIPESTATUS[0]}
set -e

summary="$(tail -n 1 /tmp/mfa-ci-pytest.log | sed 's/=//g' | xargs)"
pyver="$("$VENV/bin/python" -c 'import platform; print(platform.python_version())')"
echo "ci.sh: python ${pyver}: ${summary} (exit ${status})"

if [[ "${1:-}" == "--post-status" ]]; then
  sha="$(git rev-parse HEAD)"
  repo="$(git remote get-url origin | sed -E 's#.*github.com[:/]##; s#\.git$##; s#^git/##' )"
  state=$([[ $status -eq 0 ]] && echo success || echo failure)
  where="${CI_RUNNER_LABEL:-$(whoami)@$(hostname)}"
  if ! git diff --quiet HEAD; then
    echo "ci.sh: working tree has uncommitted changes; not posting a status for ${sha}" >&2
    exit 2
  fi
  gh api -X POST "repos/${repo}/statuses/${sha}" \
    -f state="$state" \
    -f context="local-ci (scripts/ci.sh)" \
    -f description="$(echo "py${pyver}: ${summary} — run outside GitHub Actions on ${where}" | cut -c1-140)" >/dev/null
  echo "ci.sh: posted '${state}' to ${repo}@${sha:0:7}"
fi
exit "$status"
