#!/usr/bin/env bash
set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_NAME="${RUN_NAME:-all_12288}"
LOCAL_ROOT="${LOCAL_ROOT:-runs}"
RUN_DIR="${RUN_DIR:-}"
REMOTE="${REMOTE:-}"
EPISODES_DIR_NAME="${EPISODES_DIR_NAME:-domain_teacher_episodes}"
OUTPUT_DIR_NAME="${OUTPUT_DIR_NAME:-qwen_teacher_episodes}"
SCORER_DIR_NAME="${SCORER_DIR_NAME:-teacher_scorer}"
RANKING_DIR_NAME="${RANKING_DIR_NAME:-teacher_ranked}"

DOMAIN_SET="${DOMAIN_SET:-all}"
EPISODES="${EPISODES:-12288}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-16}"
SEED="${SEED:-73037}"
SHARD_COUNT="${SHARD_COUNT:-768}"
EXPECTED_PER_SHARD="${EXPECTED_PER_SHARD:-}"
WORKERS="${WORKERS:-3}"
TARGET_COMPLETE_SHARDS="${TARGET_COMPLETE_SHARDS:-0}"
MAX_ROUNDS="${MAX_ROUNDS:-0}"
ROUND_SLEEP_SECONDS="${ROUND_SLEEP_SECONDS:-0}"

QWEN_MODEL="${QWEN_MODEL:-qwen-plus}"
QWEN_BASE_URL_VALUE="${QWEN_BASE_URL:-${DASHSCOPE_BASE_URL:-https://dashscope-intl.aliyuncs.com/compatible-mode/v1}}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-60}"
SHARD_TIMEOUT="${SHARD_TIMEOUT:-420}"
RETRIES="${RETRIES:-2}"
RETRY_DELAY="${RETRY_DELAY:-2}"

PYTHON_BIN="${PYTHON_BIN:-}"
RCLONE_BIN="${RCLONE_BIN:-rclone}"
RCLONE_TRANSFERS="${RCLONE_TRANSFERS:-8}"
RCLONE_CHECKERS="${RCLONE_CHECKERS:-16}"
RCLONE_STATS="${RCLONE_STATS:-30s}"

PULL_REMOTE="${PULL_REMOTE:-1}"
PUSH_REMOTE="${PUSH_REMOTE:-1}"
GENERATE_EPISODES="${GENERATE_EPISODES:-1}"
CONVERT_ON_COMPLETE="${CONVERT_ON_COMPLETE:-1}"
STOP_ON_NO_PROGRESS="${STOP_ON_NO_PROGRESS:-1}"
PROMPT_QWEN_KEY="${PROMPT_QWEN_KEY:-1}"
PROMPT_RCLONE_CONFIG="${PROMPT_RCLONE_CONFIG:-1}"
VALIDATE_QWEN_KEY="${VALIDATE_QWEN_KEY:-1}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/local_qwen_drive_run.sh [options]

Resume a Qwen teacher-labeling run locally while copying artifacts to Google
Drive through rclone. This does not mount Google Drive.

Common:
  --run-name NAME              Default: all_12288
  --run-dir PATH               Default: runs/<run-name>
  --remote REMOTE              Default: gdrive:hippo-qwen-runs/<run-name>
  --workers N                  Parallel local shard workers. Default: 3
  --target-complete-shards N   Stop after N complete shards. Default: 0, all shards
  --max-rounds N               Stop after N rounds. Default: 0, run until complete

Data shape:
  --episodes N                 Default: 12288
  --domain-set curated|broad|all
  --candidate-limit N          Default: 16
  --shard-count N              Default: 768
  --expected-per-shard N       Default: episodes / shard-count when divisible

Qwen:
  --model MODEL                Default: qwen-plus
  --base-url URL               Default: QWEN_BASE_URL or DashScope intl endpoint
  --request-timeout SEC        Default: 60
  --shard-timeout SEC          Default: 420
  --retries N                  Default: 2

Drive:
  --rclone-bin PATH            Default: rclone
  --no-pull                    Do not copy existing Drive artifacts down first
  --no-push                    Do not copy local artifacts up after each round
  --no-drive-prompt            Do not launch rclone config if remote is missing

Stages:
  --no-generate                Require episode files to already exist
  --no-convert                 Do not convert completed labels into train files
  --keep-going-on-no-progress  Do not stop when workers fail without adding lines
  --no-key-prompt              Require key env instead of prompting securely
  --no-key-check               Do not run a small Qwen preflight request

Environment:
  DASHSCOPE_API_KEY or QWEN_API_KEY may be set ahead of time. If neither is set
  and stdin is interactive, the script prompts for the key without echoing it.

Example:
  scripts/local_qwen_drive_run.sh \
    --run-name all_12288 \
    --remote gdrive:hippo-qwen-runs/all_12288 \
    --workers 3 \
    --target-complete-shards 256
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --local-root)
      LOCAL_ROOT="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --remote)
      REMOTE="$2"
      shift 2
      ;;
    --episodes-dir-name)
      EPISODES_DIR_NAME="$2"
      shift 2
      ;;
    --output-dir-name)
      OUTPUT_DIR_NAME="$2"
      shift 2
      ;;
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    --domain-set)
      DOMAIN_SET="$2"
      shift 2
      ;;
    --candidate-limit)
      CANDIDATE_LIMIT="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --shard-count)
      SHARD_COUNT="$2"
      shift 2
      ;;
    --expected-per-shard)
      EXPECTED_PER_SHARD="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --target-complete-shards)
      TARGET_COMPLETE_SHARDS="$2"
      shift 2
      ;;
    --max-rounds)
      MAX_ROUNDS="$2"
      shift 2
      ;;
    --round-sleep-seconds)
      ROUND_SLEEP_SECONDS="$2"
      shift 2
      ;;
    --model)
      QWEN_MODEL="$2"
      shift 2
      ;;
    --base-url)
      QWEN_BASE_URL_VALUE="$2"
      shift 2
      ;;
    --request-timeout)
      REQUEST_TIMEOUT="$2"
      shift 2
      ;;
    --shard-timeout)
      SHARD_TIMEOUT="$2"
      shift 2
      ;;
    --retries)
      RETRIES="$2"
      shift 2
      ;;
    --retry-delay)
      RETRY_DELAY="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --rclone-bin)
      RCLONE_BIN="$2"
      shift 2
      ;;
    --no-pull)
      PULL_REMOTE=0
      shift
      ;;
    --no-push)
      PUSH_REMOTE=0
      shift
      ;;
    --no-drive-prompt)
      PROMPT_RCLONE_CONFIG=0
      shift
      ;;
    --no-generate)
      GENERATE_EPISODES=0
      shift
      ;;
    --no-convert)
      CONVERT_ON_COMPLETE=0
      shift
      ;;
    --keep-going-on-no-progress)
      STOP_ON_NO_PROGRESS=0
      shift
      ;;
    --no-key-prompt)
      PROMPT_QWEN_KEY=0
      shift
      ;;
    --no-key-check)
      VALIDATE_QWEN_KEY=0
      shift
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$LOCAL_ROOT/$RUN_NAME"
fi
if [[ -z "$REMOTE" ]]; then
  REMOTE="gdrive:hippo-qwen-runs/$RUN_NAME"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
if [[ -z "$EXPECTED_PER_SHARD" ]]; then
  if (( EPISODES % SHARD_COUNT != 0 )); then
    die "--expected-per-shard is required when episodes is not divisible by shard-count"
  fi
  EXPECTED_PER_SHARD=$((EPISODES / SHARD_COUNT))
fi

EPISODES_DIR="$RUN_DIR/$EPISODES_DIR_NAME"
OUTPUT_DIR="$RUN_DIR/$OUTPUT_DIR_NAME"
SCORER_DIR="$RUN_DIR/$SCORER_DIR_NAME"
RANKING_DIR="$RUN_DIR/$RANKING_DIR_NAME"
LOG_DIR="$RUN_DIR/logs"
STATUS_FILE="$RUN_DIR/local_qwen_drive_status.json"
CONFIG_FILE="$RUN_DIR/local_qwen_drive_config.json"

require_command() {
  if command -v "$1" >/dev/null 2>&1; then
    return 0
  fi
  if [[ "$1" == "$RCLONE_BIN" && "$RCLONE_BIN" == "rclone" ]]; then
    die "missing command: rclone. On Gentoo install it with: sudo emerge --ask net-misc/rclone"
  fi
  die "missing command: $1"
}

is_interactive() {
  [[ -t 0 && -t 1 ]]
}

ensure_qwen_key() {
  if [[ -n "${DASHSCOPE_API_KEY:-${QWEN_API_KEY:-}}" ]]; then
    return 0
  fi
  if [[ "$PROMPT_QWEN_KEY" == "1" ]] && is_interactive; then
    local key
    printf "Qwen/DashScope API key: " >&2
    IFS= read -r -s key
    printf "\n" >&2
    key="${key//[[:space:]]/}"
    [[ -n "$key" ]] || die "empty Qwen key"
    export DASHSCOPE_API_KEY="$key"
    return 0
  fi
  die "set DASHSCOPE_API_KEY or QWEN_API_KEY before running local Qwen labeling"
}

describe_qwen_key() {
  local key="${DASHSCOPE_API_KEY:-${QWEN_API_KEY:-}}"
  local length="${#key}"
  local preview
  if (( length <= 8 )); then
    preview="<too-short>"
  else
    preview="${key:0:4}...${key: -4}"
  fi
  local fingerprint
  fingerprint="$(
    QWEN_KEY_FOR_HASH="$key" "$PYTHON_BIN" - <<'PY'
import hashlib
import os

key = os.environ.get("QWEN_KEY_FOR_HASH", "")
print(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12])
PY
  )"
  echo "qwen: key preview=$preview length=$length sha256_12=$fingerprint"
}

validate_qwen_key() {
  [[ "$VALIDATE_QWEN_KEY" == "1" ]] || return 0
  echo "qwen: validating API key and base URL"
  "$PYTHON_BIN" - "$QWEN_MODEL" "$QWEN_BASE_URL_VALUE" <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request

model = sys.argv[1]
base_url = sys.argv[2].rstrip("/")
api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY") or ""
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Return exactly OK."}],
    "temperature": 0,
    "max_tokens": 8,
}
request = urllib.request.Request(
    base_url + "/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8")
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="replace")
    print(f"Qwen preflight failed: HTTP {exc.code}: {body[:800]}", file=sys.stderr)
    raise SystemExit(1)
except Exception as exc:
    print(f"Qwen preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)
data = json.loads(body)
content = data["choices"][0]["message"]["content"]
print(f"qwen: preflight ok ({content[:40]!r})")
PY
}

remote_name() {
  local remote="$1"
  if [[ "$remote" != *:* ]]; then
    die "remote must look like name:path, got: $remote"
  fi
  printf "%s" "${remote%%:*}"
}

ensure_rclone_remote() {
  [[ "$PULL_REMOTE" == "1" || "$PUSH_REMOTE" == "1" ]] || return 0
  local name
  name="$(remote_name "$REMOTE")"
  if "$RCLONE_BIN" listremotes | grep -Fxq "${name}:"; then
    return 0
  fi
  if [[ "$PROMPT_RCLONE_CONFIG" == "1" ]] && is_interactive; then
    echo "rclone remote '${name}:' is not configured."
    echo "Launching rclone config. Create a Google Drive remote named '${name}'."
    "$RCLONE_BIN" config
    if "$RCLONE_BIN" listremotes | grep -Fxq "${name}:"; then
      return 0
    fi
    die "rclone remote '${name}:' is still missing after config"
  fi
  die "rclone remote '${name}:' is not configured; run 'rclone config' first"
}

if [[ "$PULL_REMOTE" == "1" || "$PUSH_REMOTE" == "1" ]]; then
  require_command "$RCLONE_BIN"
  ensure_rclone_remote
fi
require_command "$PYTHON_BIN"
ensure_qwen_key
describe_qwen_key
validate_qwen_key

mkdir -p "$RUN_DIR" "$EPISODES_DIR" "$OUTPUT_DIR" "$SCORER_DIR" "$RANKING_DIR" "$LOG_DIR"

rclone_common_args=(
  --update
  --transfers "$RCLONE_TRANSFERS"
  --checkers "$RCLONE_CHECKERS"
  --stats "$RCLONE_STATS"
  --create-empty-src-dirs
)

pull_remote() {
  [[ "$PULL_REMOTE" == "1" ]] || return 0
  echo "pull: $REMOTE -> $RUN_DIR"
  if "$RCLONE_BIN" lsf "$REMOTE" >/dev/null 2>&1; then
    "$RCLONE_BIN" copy "$REMOTE" "$RUN_DIR" "${rclone_common_args[@]}" || die "remote pull failed"
  else
    echo "pull: remote path does not exist yet; starting from local files"
  fi
}

push_remote() {
  [[ "$PUSH_REMOTE" == "1" ]] || return 0
  echo "push: $RUN_DIR -> $REMOTE"
  "$RCLONE_BIN" mkdir "$REMOTE" >/dev/null 2>&1 || true
  "$RCLONE_BIN" copy "$RUN_DIR" "$REMOTE" "${rclone_common_args[@]}" || die "remote push failed"
}

nonempty_line_count() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo 0
    return 0
  fi
  awk 'NF { count += 1 } END { print count + 0 }' "$path"
}

shard_path() {
  local shard="$1"
  printf "%s/episodes_%03d.jsonl" "$OUTPUT_DIR" "$shard"
}

shard_lines() {
  nonempty_line_count "$(shard_path "$1")"
}

shard_summary() {
  local complete=0
  local partial=0
  local missing=0
  local total_lines=0
  local shard=0
  local lines=0
  for ((shard = 0; shard < SHARD_COUNT; shard += 1)); do
    lines="$(shard_lines "$shard")"
    total_lines=$((total_lines + lines))
    if (( lines >= EXPECTED_PER_SHARD )); then
      complete=$((complete + 1))
    elif (( lines > 0 )); then
      partial=$((partial + 1))
    else
      missing=$((missing + 1))
    fi
  done
  printf "%s %s %s %s\n" "$complete" "$partial" "$missing" "$total_lines"
}

collect_next_shards() {
  local limit="$1"
  local emitted=0
  local shard=0
  local lines=0
  for ((shard = 0; shard < SHARD_COUNT; shard += 1)); do
    lines="$(shard_lines "$shard")"
    if (( lines < EXPECTED_PER_SHARD )); then
      printf "%s\n" "$shard"
      emitted=$((emitted + 1))
      if (( emitted >= limit )); then
        return 0
      fi
    fi
  done
}

write_config() {
  cat > "$CONFIG_FILE" <<EOF
{
  "run_name": "$RUN_NAME",
  "run_dir": "$RUN_DIR",
  "remote": "$REMOTE",
  "episodes": $EPISODES,
  "domain_set": "$DOMAIN_SET",
  "candidate_limit": $CANDIDATE_LIMIT,
  "seed": $SEED,
  "shard_count": $SHARD_COUNT,
  "expected_per_shard": $EXPECTED_PER_SHARD,
  "workers": $WORKERS,
  "target_complete_shards": $TARGET_COMPLETE_SHARDS,
  "qwen_model": "$QWEN_MODEL",
  "qwen_base_url": "$QWEN_BASE_URL_VALUE"
}
EOF
}

write_status() {
  local state="$1"
  local round="$2"
  local complete="$3"
  local partial="$4"
  local missing="$5"
  local total_lines="$6"
  cat > "$STATUS_FILE" <<EOF
{
  "state": "$state",
  "round": $round,
  "run_name": "$RUN_NAME",
  "run_dir": "$RUN_DIR",
  "remote": "$REMOTE",
  "output_dir": "$OUTPUT_DIR",
  "complete_shards": $complete,
  "partial_shards": $partial,
  "missing_shards": $missing,
  "total_labeled_lines": $total_lines,
  "shard_count": $SHARD_COUNT,
  "expected_per_shard": $EXPECTED_PER_SHARD,
  "target_complete_shards": $TARGET_COMPLETE_SHARDS,
  "updated_at_unix": $(date +%s)
}
EOF
}

generate_episodes_if_needed() {
  local episode_file="$EPISODES_DIR/episodes_000.jsonl"
  if [[ -s "$episode_file" ]]; then
    local lines
    lines="$(nonempty_line_count "$episode_file")"
    echo "episodes: found $episode_file lines=$lines"
    if (( lines != EPISODES )); then
      echo "warning: requested episodes=$EPISODES but existing file has lines=$lines"
    fi
    return 0
  fi
  [[ "$GENERATE_EPISODES" == "1" ]] || die "missing $episode_file and --no-generate was set"
  echo "episodes: generating $EPISODES domain episodes into $EPISODES_DIR"
  "$PYTHON_BIN" scripts/generate_domain_teacher_episodes.py \
    --domain-set "$DOMAIN_SET" \
    --episodes "$EPISODES" \
    --candidate-limit "$CANDIDATE_LIMIT" \
    --seed "$SEED" \
    --output-dir "$EPISODES_DIR" || die "episode generation failed"
}

run_shard_worker() {
  local shard="$1"
  local padded
  padded="$(printf "%03d" "$shard")"
  local log_file="$LOG_DIR/qwen_shard_${padded}_$(date +%Y%m%d_%H%M%S).log"
  echo "worker: shard=$padded log=$log_file"
  "$PYTHON_BIN" scripts/run_qwen_label_shards.py \
    --episodes-dir "$EPISODES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --shard-count "$SHARD_COUNT" \
    --start-shard "$shard" \
    --max-shards 1 \
    --expected-per-shard "$EXPECTED_PER_SHARD" \
    --shard-timeout "$SHARD_TIMEOUT" \
    --request-timeout "$REQUEST_TIMEOUT" \
    --retries "$RETRIES" \
    --retry-delay "$RETRY_DELAY" \
    --model "$QWEN_MODEL" \
    --base-url "$QWEN_BASE_URL_VALUE" \
    --continue-on-failure \
    --progress-file "$OUTPUT_DIR/progress_shard_${padded}.json" \
    > "$log_file" 2>&1
  local status=$?
  tail -n 5 "$log_file" || true
  return "$status"
}

convert_labels() {
  [[ "$CONVERT_ON_COMPLETE" == "1" ]] || return 0
  echo "convert: writing scorer data into $SCORER_DIR and $RANKING_DIR"
  "$PYTHON_BIN" scripts/convert_teacher_episodes.py \
    --episodes-dir "$OUTPUT_DIR" \
    --output-data-dir "$SCORER_DIR" \
    --output-ranking-dir "$RANKING_DIR" || die "teacher conversion failed"
}

round=0
ACTIVE_PIDS=()

handle_interrupt() {
  trap - INT TERM
  echo "interrupt: stopping workers and pushing current artifacts"
  for pid in "${ACTIVE_PIDS[@]}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
  for pid in "${ACTIVE_PIDS[@]}"; do
    wait "$pid" >/dev/null 2>&1 || true
  done
  local complete partial missing total
  read -r complete partial missing total < <(shard_summary)
  write_status "interrupted" "$round" "$complete" "$partial" "$missing" "$total"
  push_remote || true
  exit 130
}

trap handle_interrupt INT TERM

pull_remote
write_config
generate_episodes_if_needed
push_remote

while true; do
  read -r before_complete before_partial before_missing before_total < <(shard_summary)
  echo "status: complete=$before_complete/$SHARD_COUNT partial=$before_partial missing=$before_missing labeled_lines=$before_total"
  write_status "running" "$round" "$before_complete" "$before_partial" "$before_missing" "$before_total"

  if (( before_complete >= SHARD_COUNT )); then
    write_status "complete" "$round" "$before_complete" "$before_partial" "$before_missing" "$before_total"
    convert_labels
    push_remote
    echo "done: all shards complete"
    exit 0
  fi
  if (( TARGET_COMPLETE_SHARDS > 0 && before_complete >= TARGET_COMPLETE_SHARDS )); then
    write_status "target_complete" "$round" "$before_complete" "$before_partial" "$before_missing" "$before_total"
    convert_labels
    push_remote
    echo "stopped: reached target complete shards $TARGET_COMPLETE_SHARDS"
    exit 0
  fi
  if (( MAX_ROUNDS > 0 && round >= MAX_ROUNDS )); then
    push_remote
    echo "stopped: reached max rounds $MAX_ROUNDS"
    exit 0
  fi

  round_worker_limit="$WORKERS"
  if (( TARGET_COMPLETE_SHARDS > 0 )); then
    remaining_to_target=$((TARGET_COMPLETE_SHARDS - before_complete))
    if (( remaining_to_target < round_worker_limit )); then
      round_worker_limit="$remaining_to_target"
    fi
  fi

  mapfile -t shards < <(collect_next_shards "$round_worker_limit")
  if (( ${#shards[@]} == 0 )); then
    push_remote
    echo "done: no incomplete shards found"
    exit 0
  fi

  round=$((round + 1))
  echo "round $round: running shards ${shards[*]}"
  ACTIVE_PIDS=()
  for shard in "${shards[@]}"; do
    run_shard_worker "$shard" &
    ACTIVE_PIDS+=("$!")
  done

  round_failed=0
  for pid in "${ACTIVE_PIDS[@]}"; do
    if wait "$pid"; then
      :
    else
      round_failed=1
    fi
  done
  ACTIVE_PIDS=()

  read -r after_complete after_partial after_missing after_total < <(shard_summary)
  echo "round $round result: complete=$after_complete/$SHARD_COUNT partial=$after_partial missing=$after_missing labeled_lines=$after_total"
  write_status "running" "$round" "$after_complete" "$after_partial" "$after_missing" "$after_total"
  push_remote

  if (( round_failed != 0 && STOP_ON_NO_PROGRESS == 1 )); then
    if (( after_complete == before_complete && after_total == before_total )); then
      write_status "blocked" "$round" "$after_complete" "$after_partial" "$after_missing" "$after_total"
      die "workers failed without adding labeled lines; check logs in $LOG_DIR"
    fi
  fi

  if (( ROUND_SLEEP_SECONDS > 0 )); then
    sleep "$ROUND_SLEEP_SECONDS"
  fi
done
