#!/usr/bin/env bash
# Synchronize GitHub main into a GitCode fork branch as one CLA-authored commit.
# The default mode only prepares and validates a candidate. Use --push explicitly.

set -euo pipefail

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH='' cd -- "${SCRIPT_DIR}/.." && pwd)"

GITHUB_REMOTE="${PTOAS_SYNC_GITHUB_REMOTE:-hwupstream}"
GITHUB_REF="${PTOAS_SYNC_GITHUB_REF:-refs/heads/main}"
GITCODE_REMOTE="${PTOAS_SYNC_GITCODE_REMOTE:-gitcode}"
GITCODE_REF="${PTOAS_SYNC_GITCODE_REF:-refs/heads/master}"
FORK_REMOTE="${PTOAS_SYNC_FORK_REMOTE:-gitcodefork}"
SYNC_BRANCH="${PTOAS_SYNC_BRANCH:-codex/sync-github-main}"
PROTECTED_PATHS_FILE="${PTOAS_SYNC_PROTECTED_PATHS:-${SCRIPT_DIR}/sync_github_to_gitcode.paths}"
AUTHOR_NAME="${PTOAS_SYNC_AUTHOR_NAME:-hecrereed}"
AUTHOR_EMAIL="${PTOAS_SYNC_AUTHOR_EMAIL:-821896444@qq.com}"
COMMIT_SUBJECT="${PTOAS_SYNC_COMMIT_SUBJECT:-sync: update GitCode from GitHub main}"
PUSH=0
GITHUB_BASE="${PTOAS_SYNC_GITHUB_BASE:-}"
PREVIOUS_GITHUB_HEAD="${PTOAS_SYNC_PREVIOUS_GITHUB_HEAD:-}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: tools/sync_github_to_gitcode.sh [options]

Fetch GitHub main and GitCode master, build a candidate sync commit, and print
the commit SHA. The candidate is not pushed unless --push is supplied.

Options:
  --push                         Push the candidate with force-with-lease.
  --github-base SHA              Common GitHub source base for bootstrapping.
  --previous-github-head SHA    Last GitHub SHA already present in an old branch.
  --protected-paths FILE         Override the GitCode path protection list.
  -h, --help                     Show this help.

The first run against an existing squash branch made before this script needs
both --github-base and --previous-github-head. Later runs read the trailers
written by this script automatically.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --push)
      PUSH=1
      ;;
    --github-base)
      [ "$#" -ge 2 ] || die "--github-base requires a commit SHA"
      GITHUB_BASE="$2"
      shift
      ;;
    --previous-github-head)
      [ "$#" -ge 2 ] || die "--previous-github-head requires a commit SHA"
      PREVIOUS_GITHUB_HEAD="$2"
      shift
      ;;
    --protected-paths)
      [ "$#" -ge 2 ] || die "--protected-paths requires a file"
      PROTECTED_PATHS_FILE="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (use --help)"
      ;;
  esac
  shift
done

cd "$REPO_ROOT"
git rev-parse --git-dir >/dev/null 2>&1 || die "not inside a Git repository"
[ -r "$PROTECTED_PATHS_FILE" ] || die "protected path list not readable: $PROTECTED_PATHS_FILE"

remote_configured() {
  git config --get "remote.$1.url" >/dev/null 2>&1
}

remote_configured "$GITHUB_REMOTE" || die "GitHub remote is not configured: $GITHUB_REMOTE"
remote_configured "$GITCODE_REMOTE" || die "GitCode remote is not configured: $GITCODE_REMOTE"
remote_configured "$FORK_REMOTE" || die "fork remote is not configured: $FORK_REMOTE"

sync_ref="refs/heads/${SYNC_BRANCH#refs/heads/}"

remote_sha() {
  local remote="$1"
  local ref="$2"
  git ls-remote "$remote" "$ref" | awk 'NR == 1 { print $1; exit }'
}

printf 'Fetching GitHub %s and GitCode %s...\n' "$GITHUB_REF" "$GITCODE_REF"
git fetch --no-tags "$GITHUB_REMOTE" "$GITHUB_REF"
github_head="$(git rev-parse FETCH_HEAD)"
git fetch --no-tags "$GITCODE_REMOTE" "$GITCODE_REF"
gitcode_head="$(git rev-parse FETCH_HEAD)"

previous_sync="$(remote_sha "$FORK_REMOTE" "$sync_ref")"
if [ -n "$previous_sync" ]; then
  git fetch --no-tags "$FORK_REMOTE" "$sync_ref"
  git cat-file -e "${previous_sync}^{commit}" \
    || die "cannot read existing sync branch commit: $previous_sync"
fi

trailer_from_commit() {
  local key="$1"
  local commit="$2"
  git show -s --format="%(trailers:key=${key},valueonly)" "$commit" \
    | awk 'NF { value = $0 } END { print value }'
}

source_base="${GITHUB_BASE}"
if [ -n "$previous_sync" ]; then
  previous_source_base="$(trailer_from_commit GitHub-Source-Base "$previous_sync")"
  [ -n "$previous_source_base" ] && source_base="$previous_source_base"
  previous_trailer_head="$(trailer_from_commit GitHub-Head "$previous_sync")"
  [ -n "$previous_trailer_head" ] && PREVIOUS_GITHUB_HEAD="$previous_trailer_head"
fi

if [ -n "$previous_sync" ] && [ -z "$PREVIOUS_GITHUB_HEAD" ]; then
  die "existing sync branch has no GitHub-Head trailer; pass --previous-github-head and --github-base once"
fi

if [ -z "$PREVIOUS_GITHUB_HEAD" ]; then
  [ -n "$source_base" ] || die "bootstrap requires --github-base (the common GitHub source base)"
  PREVIOUS_GITHUB_HEAD="$source_base"
fi

[ -n "$source_base" ] || source_base="$PREVIOUS_GITHUB_HEAD"
git cat-file -e "${PREVIOUS_GITHUB_HEAD}^{commit}" \
  || die "previous GitHub head is not available locally: $PREVIOUS_GITHUB_HEAD"
git cat-file -e "${source_base}^{commit}" \
  || die "GitHub source base is not available locally: $source_base"

git merge-base --is-ancestor "$PREVIOUS_GITHUB_HEAD" "$github_head" \
  || die "GitHub main is not a descendant of the previous imported head; inspect a rewritten history manually"

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/ptoas-gitcode-sync.XXXXXX")"
tmp_worktree="${tmp_root}/worktree"
cleanup() {
  git worktree remove --force "$tmp_worktree" >/dev/null 2>&1 || true
  rmdir "$tmp_root" >/dev/null 2>&1 || true
}
trap cleanup EXIT HUP INT TERM

if [ -n "$previous_sync" ]; then
  git worktree add --detach "$tmp_worktree" "$previous_sync" >/dev/null
else
  git worktree add --detach "$tmp_worktree" "$gitcode_head" >/dev/null
fi

protected_paths=()
while IFS= read -r raw_path || [ -n "$raw_path" ]; do
  path="${raw_path#"${raw_path%%[![:space:]]*}"}"
  path="${path%"${path##*[![:space:]]}"}"
  [ -n "$path" ] || continue
  case "$path" in
    \#*) continue ;;
  esac
  protected_paths+=("$path")
done < "$PROTECTED_PATHS_FILE"

is_protected() {
  local candidate="$1"
  local protected
  for protected in "${protected_paths[@]}"; do
    case "$candidate" in
      "$protected"|"$protected"/*)
        return 0
        ;;
    esac
  done
  return 1
}

path_exists_in_tree() {
  local tree="$1"
  local path="$2"
  git -C "$tmp_worktree" ls-tree -r --name-only "$tree" -- "$path" \
    | grep -q .
}

restore_protected_path() {
  local path="$1"
  if path_exists_in_tree "$gitcode_head" "$path"; then
    git -C "$tmp_worktree" checkout "$gitcode_head" -- "$path"
    git -C "$tmp_worktree" add -A -- "$path"
  else
    git -C "$tmp_worktree" rm -r -f --ignore-unmatch -- "$path" >/dev/null 2>&1 || true
  fi
}

resolve_gitcode_conflicts() {
  local conflicts
  local path
  conflicts="$(git -C "$tmp_worktree" diff --name-only --diff-filter=U)"
  [ -n "$conflicts" ] || return 0
  while IFS= read -r path; do
    [ -n "$path" ] || continue
    if is_protected "$path"; then
      printf 'Resolving protected path from GitCode master: %s\n' "$path"
      restore_protected_path "$path"
    else
      printf 'Unresolved non-protected conflict: %s\n' "$path" >&2
    fi
  done <<EOF
$conflicts
EOF
  [ -z "$(git -C "$tmp_worktree" diff --name-only --diff-filter=U)" ] \
    || die "manual resolution required; no candidate was pushed"
}

if [ -n "$previous_sync" ]; then
  printf 'Rebasing the previous sync tree onto GitCode master %s...\n' "$gitcode_head"
  if ! git -C "$tmp_worktree" merge --no-commit --no-ff "$gitcode_head"; then
    [ -n "$(git -C "$tmp_worktree" diff --name-only --diff-filter=U)" ] \
      || die "merging GitCode master failed for a reason other than a file conflict"
    resolve_gitcode_conflicts
  fi
fi

sync_tree="$(git -C "$tmp_worktree" write-tree)"
synthetic_parent="$(
  GIT_AUTHOR_NAME=ptoas-sync \
  GIT_AUTHOR_EMAIL=ptoas-sync@localhost \
  GIT_COMMITTER_NAME=ptoas-sync \
  GIT_COMMITTER_EMAIL=ptoas-sync@localhost \
  git commit-tree "$sync_tree" -p "$PREVIOUS_GITHUB_HEAD" -m 'temporary PTOAS sync ancestor'
)"
git -C "$tmp_worktree" reset --hard "$synthetic_parent" >/dev/null

printf 'Applying GitHub changes %s..%s...\n' "$PREVIOUS_GITHUB_HEAD" "$github_head"
if ! git -C "$tmp_worktree" merge --no-commit --no-ff "$github_head"; then
  [ -n "$(git -C "$tmp_worktree" diff --name-only --diff-filter=U)" ] \
    || die "merging GitHub main failed for a reason other than a file conflict"
  resolve_gitcode_conflicts
fi

# Restore protected files even when Git happened to merge their hunks cleanly.
for path in "${protected_paths[@]}"; do
  restore_protected_path "$path"
done

[ -z "$(git -C "$tmp_worktree" diff --name-only --diff-filter=U)" ] \
  || die "candidate still contains unmerged paths"

runop_file="$tmp_worktree/test/samples/runop.sh"
if [ -f "$runop_file" ]; then
  grep -q 'PTOAS_PRESMOKE_SKIP_RUNOP' "$runop_file" \
    || die "runop.sh lost the GitCode PreSmoke skip marker"
  grep -q 'PTOAS_SAMPLE_JOBS' "$runop_file" \
    || die "runop.sh lost the GitHub sample parallelism setting"
fi

candidate_tree="$(git -C "$tmp_worktree" write-tree)"
for path in "${protected_paths[@]}"; do
  if ! git diff --quiet "$gitcode_head" "$candidate_tree" -- "$path"; then
    die "protected path differs from GitCode master after merge: $path"
  fi
done

if [ -n "$previous_sync" ] && git diff --quiet "$previous_sync" "$candidate_tree"; then
  printf '\nGitCode sync branch is already up to date; no push is needed.\n'
  exit 0
fi

commit_message="$(printf '%s\n\n%s\n\nGitHub-Source-Base: %s\nGitHub-Previous-Head: %s\nGitHub-Head: %s\nGitCode-Base: %s\n' \
  "$COMMIT_SUBJECT" \
  'Synchronize GitCode with GitHub main while retaining GitCode-specific build, packaging, and CI files.' \
  "$source_base" \
  "$PREVIOUS_GITHUB_HEAD" \
  "$github_head" \
  "$gitcode_head")"
final_commit="$(
  printf '%s' "$commit_message" |
  GIT_AUTHOR_NAME="$AUTHOR_NAME" \
  GIT_AUTHOR_EMAIL="$AUTHOR_EMAIL" \
  GIT_COMMITTER_NAME="$AUTHOR_NAME" \
  GIT_COMMITTER_EMAIL="$AUTHOR_EMAIL" \
  git commit-tree "$candidate_tree" -p "$gitcode_head"
)"

parent_count="$(git rev-list --parents -n 1 "$final_commit" | awk '{ print NF - 1 }')"
[ "$parent_count" = 1 ] || die "candidate does not have exactly one parent"
[ "$(git rev-parse "$final_commit^")" = "$gitcode_head" ] \
  || die "candidate parent is not the fetched GitCode master"

printf '\nCandidate commit: %s\n' "$final_commit"
printf 'Parent:           %s\n' "$gitcode_head"
printf 'GitHub range:     %s..%s\n' "$PREVIOUS_GITHUB_HEAD" "$github_head"
printf 'Author:            %s <%s>\n' "$AUTHOR_NAME" "$AUTHOR_EMAIL"
printf 'Protected paths:   %s\n' "${#protected_paths[@]}"
printf 'Changed files:     %s\n' "$(git diff --name-only "$gitcode_head" "$candidate_tree" | wc -l | tr -d ' ')"

if [ "$PUSH" -eq 1 ]; then
  printf 'Pushing %s with force-with-lease...\n' "$sync_ref"
  git push \
    --force-with-lease="${sync_ref}:${previous_sync}" \
    "$FORK_REMOTE" \
    "${final_commit}:${sync_ref}"
  printf 'Updated remote branch: %s/%s\n' "$FORK_REMOTE" "$SYNC_BRANCH"
else
  printf 'Dry-run only; pass --push to update the GitCode fork branch.\n'
fi
