#!/usr/bin/env bash
set -euo pipefail

ROOT="${WEBARENA_ROOT:-/data/wf/sxh}"
SERVER="${WEBARENA_SERVER:-10.2.131.41}"
SETUP_REPO="${ROOT}/webarena-setup"
SETUP_DIR="${SETUP_REPO}/webarena"
ARCHIVES="${ROOT}/webarena_archives"
ENV_DIR="${ROOT}/webarena_env"
DOCKER_DIR="${ROOT}/webarena_docker"
DOCKER_SOCK="${DOCKER_DIR}/docker.sock"
CONTAINERD_DIR="${ROOT}/webarena_containerd"
CONTAINERD_SOCK="${CONTAINERD_DIR}/containerd.sock"
export DOCKER_HOST="unix://${DOCKER_SOCK}"

SHOPPING_PORT="${SHOPPING_PORT:-7770}"
SHOPPING_ADMIN_PORT="${SHOPPING_ADMIN_PORT:-7780}"
REDDIT_PORT="${REDDIT_PORT:-9999}"
GITLAB_PORT="${GITLAB_PORT:-8023}"
WIKIPEDIA_PORT="${WIKIPEDIA_PORT:-8888}"
MAP_PORT="${MAP_PORT:-13000}"
HOMEPAGE_PORT="${HOMEPAGE_PORT:-4399}"

log() {
  printf '[webarena-env] %s\n' "$*"
}

sudo_do() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  elif [[ -n "${SUDO_PASSWORD:-}" ]]; then
    printf '%s\n' "${SUDO_PASSWORD}" | sudo -S "$@"
  else
    sudo "$@"
  fi
}

ensure_second_docker() {
  mkdir -p "${DOCKER_DIR}"
  if [[ -S "${DOCKER_SOCK}" ]] && docker info >/dev/null 2>&1 \
    && pgrep -af "dockerd .*--containerd ${CONTAINERD_SOCK}" >/dev/null; then
    log "second Docker daemon already running at ${DOCKER_SOCK}"
    return
  fi

  if [[ -f "${DOCKER_DIR}/dockerd.pid" ]]; then
    sudo_do kill "$(cat "${DOCKER_DIR}/dockerd.pid")" >/dev/null 2>&1 || true
    sleep 2
  fi

  log "starting dedicated containerd at ${CONTAINERD_SOCK}"
  mkdir -p "${CONTAINERD_DIR}"
  if ! [[ -S "${CONTAINERD_SOCK}" ]] || ! sudo_do ctr --address "${CONTAINERD_SOCK}" version >/dev/null 2>&1; then
    sudo_do bash -c "nohup setsid containerd \
      --root '${CONTAINERD_DIR}/root' \
      --state '${CONTAINERD_DIR}/state' \
      --address '${CONTAINERD_SOCK}' \
      > '${CONTAINERD_DIR}/containerd.log' 2>&1 < /dev/null &"
    for _ in $(seq 1 60); do
      if [[ -S "${CONTAINERD_SOCK}" ]] && sudo_do ctr --address "${CONTAINERD_SOCK}" version >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
  fi

  log "starting second Docker daemon with data-root ${DOCKER_DIR}/data"
  cat > "${DOCKER_DIR}/daemon.json" <<'EOF'
{
  "registry-mirrors": []
}
EOF
  sudo_do bash -c "
    ip link show webarena0 >/dev/null 2>&1 || ip link add name webarena0 type bridge
    ip addr show webarena0 | grep -q 172.29.0.1 || ip addr add 172.29.0.1/16 dev webarena0
    ip link set webarena0 up
  "
  rm -f "${DOCKER_SOCK}" "${DOCKER_DIR}/dockerd.pid"
  sudo_do bash -c "nohup setsid dockerd \
    --data-root '${DOCKER_DIR}/data' \
    --exec-root '${DOCKER_DIR}/exec' \
    --pidfile '${DOCKER_DIR}/dockerd.pid' \
    --config-file '${DOCKER_DIR}/daemon.json' \
    --host 'unix://${DOCKER_SOCK}' \
    --containerd '${CONTAINERD_SOCK}' \
    --bridge webarena0 \
    --fixed-cidr 172.29.0.0/16 \
    > '${DOCKER_DIR}/dockerd.log' 2>&1 < /dev/null &"

  for _ in $(seq 1 60); do
    if [[ -S "${DOCKER_SOCK}" ]]; then
      sudo_do chown "$(id -u):$(id -g)" "${DOCKER_SOCK}" || true
    fi
    if docker info >/dev/null 2>&1; then
      log "second Docker daemon is ready"
      return
    fi
    sleep 1
  done

  tail -120 "${DOCKER_DIR}/dockerd.log" || true
  log "failed to start second Docker daemon"
  exit 1
}

ensure_setup_repo() {
  mkdir -p "${ROOT}"
  if [[ ! -d "${SETUP_REPO}/.git" ]]; then
    git clone https://github.com/gasse/webarena-setup.git "${SETUP_REPO}"
  fi
}

write_vars() {
  cat > "${SETUP_DIR}/00_vars.sh" <<EOF
#!/bin/bash
PUBLIC_HOSTNAME="${SERVER}"
SHOPPING_PORT=${SHOPPING_PORT}
SHOPPING_ADMIN_PORT=${SHOPPING_ADMIN_PORT}
REDDIT_PORT=${REDDIT_PORT}
GITLAB_PORT=${GITLAB_PORT}
WIKIPEDIA_PORT=${WIKIPEDIA_PORT}
MAP_PORT=${MAP_PORT}
HOMEPAGE_PORT=${HOMEPAGE_PORT}
RESET_PORT=7565
SHOPPING_URL="http://\${PUBLIC_HOSTNAME}:\${SHOPPING_PORT}"
SHOPPING_ADMIN_URL="http://\${PUBLIC_HOSTNAME}:\${SHOPPING_ADMIN_PORT}/admin"
REDDIT_URL="http://\${PUBLIC_HOSTNAME}:\${REDDIT_PORT}/forums/all"
GITLAB_URL="http://\${PUBLIC_HOSTNAME}:\${GITLAB_PORT}/explore"
WIKIPEDIA_URL="http://\${PUBLIC_HOSTNAME}:\${WIKIPEDIA_PORT}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
MAP_URL="http://\${PUBLIC_HOSTNAME}:\${MAP_PORT}"
ARCHIVES_LOCATION="${ARCHIVES}"
EOF
}

write_site_env() {
  mkdir -p "${ENV_DIR}"
  cat > "${ENV_DIR}/site_env.sh" <<EOF
SHOPPING=http://${SERVER}:${SHOPPING_PORT}
SHOPPING_ADMIN=http://${SERVER}:${SHOPPING_ADMIN_PORT}/admin
REDDIT=http://${SERVER}:${REDDIT_PORT}
GITLAB=http://${SERVER}:${GITLAB_PORT}
MAP=http://${SERVER}:${MAP_PORT}
WIKIPEDIA=http://${SERVER}:${WIKIPEDIA_PORT}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing
HOMEPAGE=http://${SERVER}:${HOMEPAGE_PORT}
CLASSIFIEDS=http://${SERVER}:9980
CLASSIFIEDS_RESET_TOKEN=4b61655535e7ed388f0d40a93600254c
EOF
}

download_if_missing() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "${out}")"
  if [[ -s "${out}" ]]; then
    log "already downloaded: ${out}"
    return
  fi
  log "downloading ${url}"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 8 -s 8 -c -o "$(basename "${out}")" -d "$(dirname "${out}")" "${url}"
  else
    wget -c -O "${out}" "${url}"
  fi
}

image_exists() {
  docker image inspect "$1" >/dev/null 2>&1
}

pull_or_tag_image() {
  local source="$1"
  local target="$2"
  if image_exists "${target}"; then
    log "image exists: ${target}"
    return
  fi
  log "pulling ${source}"
  docker pull "${source}"
  docker tag "${source}" "${target}"
}

load_image_if_missing() {
  local image="$1"
  local archive="$2"
  if image_exists "${image}"; then
    log "image exists: ${image}"
    return
  fi
  log "loading ${image} from ${archive}"
  docker load --input "${archive}"
}

prepare_assets() {
  mkdir -p "${ARCHIVES}"

  pull_or_tag_image webarenaimages/shopping_final_0712:latest shopping_final_0712:latest
  pull_or_tag_image webarenaimages/shopping_admin_final_0719:latest shopping_admin_final_0719:latest
  pull_or_tag_image webarenaimages/postmill-populated-exposed-withimg:latest postmill-populated-exposed-withimg:latest
  pull_or_tag_image webarenaimages/gitlab-populated-final:latest gitlab-populated-final-port8023:latest
  docker pull ghcr.io/kiwix/kiwix-serve:3.3.0

  download_if_missing \
    http://metis.lti.cs.cmu.edu/webarena-images/wikipedia_en_all_maxi_2022-05.zim \
    "${ARCHIVES}/wikipedia_en_all_maxi_2022-05.zim"

  download_if_missing \
    https://zenodo.org/records/12636845/files/openstreetmap-website-db.tar.gz \
    "${ARCHIVES}/openstreetmap-website-db.tar.gz"
  download_if_missing \
    https://zenodo.org/records/12636845/files/openstreetmap-website-web.tar.gz \
    "${ARCHIVES}/openstreetmap-website-web.tar.gz"
  download_if_missing \
    https://zenodo.org/records/12636845/files/openstreetmap-website.tar.gz \
    "${ARCHIVES}/openstreetmap-website.tar.gz"

  load_image_if_missing openstreetmap-website-db:latest "${ARCHIVES}/openstreetmap-website-db.tar.gz"
  load_image_if_missing openstreetmap-website-web:latest "${ARCHIVES}/openstreetmap-website-web.tar.gz"

  if [[ ! -d "${SETUP_DIR}/openstreetmap-website" ]]; then
    log "extracting openstreetmap-website"
    tar -xzf "${ARCHIVES}/openstreetmap-website.tar.gz" -C "${SETUP_DIR}"
  fi

  mkdir -p "${SETUP_DIR}/wiki"
  if [[ ! -f "${SETUP_DIR}/wiki/wikipedia_en_all_maxi_2022-05.zim" ]]; then
    ln "${ARCHIVES}/wikipedia_en_all_maxi_2022-05.zim" \
      "${SETUP_DIR}/wiki/wikipedia_en_all_maxi_2022-05.zim"
  fi
}

remove_existing() {
  cd "${SETUP_DIR}"
  docker stop shopping_admin forum gitlab shopping wikipedia openstreetmap-website-db-1 openstreetmap-website-web-1 >/dev/null 2>&1 || true
  docker rm shopping_admin forum gitlab shopping wikipedia openstreetmap-website-db-1 openstreetmap-website-web-1 >/dev/null 2>&1 || true
}

create_start_patch() {
  cd "${SETUP_DIR}"
  export OSTYPE="${OSTYPE:-linux-gnu}"
  bash 03_docker_create_containers.sh
  bash 04_docker_start_containers.sh
  bash 05_docker_patch_containers.sh
}

serve_homepage() {
  cd "${SETUP_DIR}"
  python3 -m venv homepage-venv
  homepage-venv/bin/pip install -q flask
  cd webarena-homepage
  cp templates/index.backup templates/index.html
  sed -i "s|SHOPPING_URL|http://${SERVER}:${SHOPPING_PORT}|g" templates/index.html
  sed -i "s|SHOPPING_ADMIN_URL|http://${SERVER}:${SHOPPING_ADMIN_PORT}/admin|g" templates/index.html
  sed -i "s|GITLAB_URL|http://${SERVER}:${GITLAB_PORT}/explore|g" templates/index.html
  sed -i "s|REDDIT_URL|http://${SERVER}:${REDDIT_PORT}/forums/all|g" templates/index.html
  sed -i "s|MAP_URL|http://${SERVER}:${MAP_PORT}|g" templates/index.html
  sed -i "s|WIKIPEDIA_URL|http://${SERVER}:${WIKIPEDIA_PORT}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing|g" templates/index.html
  pkill -f "flask run --host=0.0.0.0 --port=${HOMEPAGE_PORT}" >/dev/null 2>&1 || true
  nohup ../homepage-venv/bin/flask --app app run --host=0.0.0.0 --port="${HOMEPAGE_PORT}" \
    > "${ENV_DIR}/homepage.log" 2>&1 &
}

health_check() {
  local urls=(
    "SHOPPING http://${SERVER}:${SHOPPING_PORT}"
    "SHOPPING_ADMIN http://${SERVER}:${SHOPPING_ADMIN_PORT}/admin"
    "REDDIT http://${SERVER}:${REDDIT_PORT}"
    "GITLAB http://${SERVER}:${GITLAB_PORT}/explore"
    "MAP http://${SERVER}:${MAP_PORT}"
    "WIKIPEDIA http://${SERVER}:${WIKIPEDIA_PORT}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
    "HOMEPAGE http://${SERVER}:${HOMEPAGE_PORT}"
  )
  for item in "${urls[@]}"; do
    local name="${item%% *}"
    local url="${item#* }"
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "${url}" || true)"
    printf '%-15s %s %s\n' "${name}" "${code}" "${url}"
  done
}

main() {
  ensure_second_docker
  if [[ "${WEBARENA_DAEMON_ONLY:-0}" == "1" ]]; then
    return
  fi
  ensure_setup_repo
  write_vars
  write_site_env
  prepare_assets
  remove_existing
  create_start_patch
  serve_homepage
  log "health check"
  health_check
}

main "$@"
