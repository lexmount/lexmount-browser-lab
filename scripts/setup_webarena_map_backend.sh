#!/usr/bin/env bash
set -euo pipefail

ROOT="${WEBARENA_ROOT:-/data/wf/sxh}"
SERVER="${WEBARENA_SERVER:-10.2.131.41}"
MAP_ROOT="${ROOT}/webarena_map_backend"
DOWNLOADS="${MAP_ROOT}/downloads"
LOGS="${MAP_ROOT}/logs"
MARKERS="${MAP_ROOT}/markers"
TMPDIR="${MAP_ROOT}/tmp"
DOCKER_DIR="${ROOT}/webarena_docker"
DOCKER_SOCKET="${DOCKER_DIR}/docker.sock"
DOCKER_VOLUME_ROOT="${DOCKER_DIR}/data/volumes"
MAP_FRONTEND_DIR="${ROOT}/webarena-setup/webarena/openstreetmap-website"
SITE_ENV="${ROOT}/webarena_env/site_env.sh"
S3_BASE="https://webarena-map-server-data.s3.us-east-1.amazonaws.com"
MIN_FREE_GIB="${WEBARENA_MAP_MIN_FREE_GIB:-450}"
MIN_AVAILABLE_MEMORY_GIB="${WEBARENA_MAP_MIN_MEMORY_GIB:-24}"
CONTAINER_MEMORY_LIMIT="30g"
OSRM_MEMORY_LIMIT="8g"
OSRM_CAR_PORT="${WEBARENA_OSRM_CAR_PORT:-15000}"
OSRM_BIKE_PORT="${WEBARENA_OSRM_BIKE_PORT:-15001}"
OSRM_FOOT_PORT="${WEBARENA_OSRM_FOOT_PORT:-15002}"
MAP_TILE_PATH="${WEBARENA_MAP_TILE_PATH:-10/284/385}"

export DOCKER_HOST="unix://${DOCKER_SOCKET}"
export TMPDIR
export XDG_CACHE_HOME="${MAP_ROOT}/cache"

declare -A EXPECTED_BYTES=(
  [osm_tile_server.tar]=41280327680
  [nominatim_volumes.tar]=124774901760
  [osm_dump.tar]=1878691840
  [osrm_routing.tar]=21278935040
)

log() {
  printf '[webarena-map] %s\n' "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

ensure_image() {
  local image="$1" attempt
  if docker image inspect "${image}" >/dev/null 2>&1; then
    log "using cached image: ${image}"
    return
  fi
  for attempt in $(seq 1 5); do
    if docker pull "${image}"; then
      return
    fi
    log "image pull failed (${attempt}/5): ${image}"
    sleep 10
  done
  die "unable to pull image: ${image}"
}

require_data_root() {
  case "${ROOT}" in
    /data/*) ;;
    *) die "WEBARENA_ROOT must be under /data; got ${ROOT}" ;;
  esac
}

print_plan() {
  cat <<EOF
root=${ROOT}
map_root=${MAP_ROOT}
downloads=${DOWNLOADS}
docker_socket=${DOCKER_SOCKET}
docker_volume_root=${DOCKER_VOLUME_ROOT}
map_frontend=${MAP_FRONTEND_DIR}
tile_url=http://${SERVER}:8080/tile/${MAP_TILE_PATH}.png
nominatim_url=http://${SERVER}:8085/
osrm_ports=${OSRM_CAR_PORT},${OSRM_BIKE_PORT},${OSRM_FOOT_PORT}
archive_bytes=189212856320
container_memory_limit=${CONTAINER_MEMORY_LIMIT}
EOF
}

guard_resources() {
  mkdir -p "${DOWNLOADS}" "${LOGS}" "${MARKERS}" "${TMPDIR}" "${XDG_CACHE_HOME}"

  local free_gib available_gib docker_root
  free_gib=$(df -Pk /data | awk 'NR == 2 {print int($4 / 1024 / 1024)}')
  ((free_gib >= MIN_FREE_GIB)) || die "/data needs ${MIN_FREE_GIB}GiB free; found ${free_gib}GiB"

  available_gib=$(awk '/MemAvailable:/ {print int($2 / 1024 / 1024)}' /proc/meminfo)
  ((available_gib >= MIN_AVAILABLE_MEMORY_GIB)) || \
    die "need ${MIN_AVAILABLE_MEMORY_GIB}GiB available memory; found ${available_gib}GiB"

  [[ -S "${DOCKER_SOCKET}" ]] || die "dedicated Docker socket is missing: ${DOCKER_SOCKET}"
  docker_root=$(docker info --format '{{.DockerRootDir}}')
  [[ "${docker_root}" == "${DOCKER_DIR}/data" ]] || \
    die "refusing Docker root outside ${DOCKER_DIR}/data: ${docker_root}"

  log "resource guard passed: /data=${free_gib}GiB free, memory=${available_gib}GiB available"
}

download_one() {
  local name="$1" expected="${EXPECTED_BYTES[$1]}" target="${DOWNLOADS}/$1"
  local actual=0 connections=4
  [[ "${name}" == "nominatim_volumes.tar" ]] && connections=16
  [[ -f "${target}" ]] && actual=$(stat -c '%s' "${target}")
  if [[ "${actual}" == "${expected}" && ! -f "${target}.aria2" ]]; then
    log "archive verified: ${name} (${actual} bytes)"
    return
  fi

  log "downloading/resuming ${name}"
  aria2c -c -x "${connections}" -s "${connections}" --min-split-size=16M \
    --file-allocation=none --auto-file-renaming=false \
    -d "${DOWNLOADS}" -o "${name}" "${S3_BASE}/${name}" \
    >"${LOGS}/${name}.log" 2>&1
  actual=$(stat -c '%s' "${target}")
  [[ "${actual}" == "${expected}" ]] || \
    die "archive size mismatch for ${name}: expected ${expected}, got ${actual}"
}

download_archives() {
  local name
  for name in osm_tile_server.tar nominatim_volumes.tar osm_dump.tar osrm_routing.tar; do
    download_one "${name}"
  done
  touch "${MARKERS}/downloads.complete"
}

extract_volume_archive() {
  local name="$1" marker="${MARKERS}/$2" strip_components="$3"
  [[ -f "${marker}" ]] && { log "already extracted: ${name}"; return; }

  log "extracting ${name} into dedicated Docker volumes"
  docker run --rm \
    --memory=1g --memory-swap=1g --cpus=2 \
    -v "${DOWNLOADS}:/archives:ro" \
    -v "${DOCKER_VOLUME_ROOT}:/target" \
    alpine:3.20 \
    tar -C /target --strip-components="${strip_components}" -xf "/archives/${name}"
  touch "${marker}"
}

extract_host_archive() {
  local name="$1" target="$2" marker="${MARKERS}/$3"
  [[ -f "${marker}" ]] && { log "already extracted: ${name}"; return; }

  mkdir -p "${target}"
  log "extracting ${name} into ${target}"
  tar -C "${target}" -xf "${DOWNLOADS}/${name}"
  touch "${marker}"
}

extract_archives() {
  ensure_image alpine:3.20
  docker volume create osm-data >/dev/null
  docker volume create osm-tiles >/dev/null
  docker volume create nominatim-data >/dev/null
  docker volume create nominatim-flatnode >/dev/null

  # The official archives were captured from hosts with different directory
  # prefixes. Preserve the Docker volume name and its _data directory.
  extract_volume_archive osm_tile_server.tar tile.extracted 4
  extract_volume_archive nominatim_volumes.tar nominatim.extracted 5
  extract_host_archive osm_dump.tar "${MAP_ROOT}/osm_dump" osm_dump.extracted
  extract_host_archive osrm_routing.tar "${MAP_ROOT}/osrm" osrm.extracted
}

remove_container() {
  docker rm -f "$1" >/dev/null 2>&1 || true
}

require_free_port() {
  local port="$1"
  if ss -ltnH "sport = :${port}" | grep -q .; then
    die "TCP port ${port} is already in use; refusing to stop an unrelated service"
  fi
}

start_backends() {
  ensure_image overv/openstreetmap-tile-server
  ensure_image mediagis/nominatim:4.2
  ensure_image ghcr.io/project-osrm/osrm-backend:v5.27.1

  local name
  for name in tile nominatim osrm-car osrm-bike osrm-foot; do
    remove_container "${name}"
  done
  for port in 8080 8085 "${OSRM_CAR_PORT}" "${OSRM_BIKE_PORT}" "${OSRM_FOOT_PORT}"; do
    require_free_port "${port}"
  done

  docker run --name tile --restart unless-stopped \
    --memory=2g --memory-swap=2g --cpus=2 \
    --volume=osm-data:/data/database/ --volume=osm-tiles:/data/tiles/ \
    -p 8080:80 -d overv/openstreetmap-tile-server run >/dev/null

  docker run --name nominatim --restart unless-stopped \
    --memory=4g --memory-swap=4g --cpus=4 \
    --env=IMPORT_STYLE=extratags \
    --env=PBF_PATH=/nominatim/data/us-northeast-latest.osm.pbf \
    --env=IMPORT_WIKIPEDIA=/nominatim/data/wikimedia-importance.sql.gz \
    --volume="${MAP_ROOT}/osm_dump/osm_dump:/nominatim/data" \
    --volume=nominatim-data:/var/lib/postgresql/14/main \
    --volume=nominatim-flatnode:/nominatim/flatnode \
    -p 8085:8080 -d mediagis/nominatim:4.2 \
    bash -lc "printf \"pg_ctl_options = '-t 600'\\n\" > /etc/postgresql/14/main/pg_ctl.conf; exec /app/start.sh" \
    >/dev/null

  local profile port
  for profile in car bike foot; do
    case "${profile}" in
      car) port="${OSRM_CAR_PORT}" ;;
      bike) port="${OSRM_BIKE_PORT}" ;;
      foot) port="${OSRM_FOOT_PORT}" ;;
    esac
    docker run --name "osrm-${profile}" --restart unless-stopped \
      --memory="${OSRM_MEMORY_LIMIT}" --memory-swap="${OSRM_MEMORY_LIMIT}" --cpus=2 \
      --volume="${MAP_ROOT}/osrm/${profile}:/data:ro" -p "${port}:5000" -d \
      ghcr.io/project-osrm/osrm-backend:v5.27.1 \
      osrm-routed --algorithm mld /data/us-northeast-latest.osrm >/dev/null
  done
}

wait_http() {
  local name="$1" url="$2" attempts="${3:-60}" code attempt
  for attempt in $(seq 1 "${attempts}"); do
    code=$(curl -sS -o /dev/null --connect-timeout 2 --max-time 10 -w '%{http_code}' "${url}" || true)
    if [[ "${code}" =~ ^[23][0-9][0-9]$ ]]; then
      log "healthy: ${name} (${code}) ${url}"
      return
    fi
    sleep 10
  done
  die "health check failed: ${name} ${url}"
}

patch_frontend() {
  [[ -d "${MAP_FRONTEND_DIR}" ]] || die "map frontend is missing: ${MAP_FRONTEND_DIR}"

  python3 - "${MAP_FRONTEND_DIR}" "${SERVER}" \
    "${OSRM_CAR_PORT}" "${OSRM_BIKE_PORT}" "${OSRM_FOOT_PORT}" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
server = sys.argv[2]
car_port, bike_port, foot_port = sys.argv[3:6]

leaflet = root / "vendor/assets/leaflet/leaflet.osm.js"
source = leaflet.read_text(encoding="utf-8")
source = source.replace(
    "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    f"http://{server}:8080/tile/{{z}}/{{x}}/{{y}}.png",
)
leaflet.write_text(source, encoding="utf-8")

settings = root / "config/settings.local.yml"
settings.write_text(
    f'nominatim_url: "http://{server}:8085/"\n'
    f'fossgis_osrm_url: "http://{server}"\n',
    encoding="utf-8",
)

directions = root / "app/assets/javascripts/index/directions/fossgis_osrm.js"
source = directions.read_text(encoding="utf-8")
source = re.sub(r'"car": "(?:/routed-car|:\d+)"', f'"car": ":{car_port}"', source)
source = re.sub(r'"bike": "(?:/routed-bike|:\d+)"', f'"bike": ":{bike_port}"', source)
source = re.sub(r'"foot": "(?:/routed-foot|:\d+)"', f'"foot": ":{foot_port}"', source)
directions.write_text(source, encoding="utf-8")

controller = root / "app/controllers/application_controller.rb"
source = controller.read_text(encoding="utf-8")
needle = ":connect_src => [Settings.nominatim_url,"
replacement = f':connect_src => ["http://{server}:*", Settings.nominatim_url,'
if replacement not in source:
    source = source.replace(needle, replacement, 1)
controller.write_text(source, encoding="utf-8")

csp = root / "config/initializers/secure_headers.rb"
source = csp.read_text(encoding="utf-8")
token = "tile.openstreetmap.org"
replacement = f"{server}:8080 {token}"
if replacement not in source:
    source = source.replace(token, replacement, 1)
csp.write_text(source, encoding="utf-8")
PY

  docker exec openstreetmap-website-web-1 sh -lc 'rm -rf /app/tmp/cache/assets /app/tmp/cache/bootsnap*'
  docker restart openstreetmap-website-web-1 >/dev/null

  mkdir -p "$(dirname "${SITE_ENV}")"
  if grep -q '^MAP_TILE=' "${SITE_ENV}" 2>/dev/null; then
    sed -i "s|^MAP_TILE=.*|MAP_TILE=http://${SERVER}:8080/tile/${MAP_TILE_PATH}.png|" "${SITE_ENV}"
  else
    printf 'MAP_TILE=http://%s:8080/tile/%s.png\n' "${SERVER}" "${MAP_TILE_PATH}" >>"${SITE_ENV}"
  fi
}

verify_all() {
  wait_http tile "http://${SERVER}:8080/tile/${MAP_TILE_PATH}.png" 90
  wait_http nominatim "http://${SERVER}:8085/search?q=Pittsburgh&format=json&limit=1" 90
  wait_http osrm-car "http://${SERVER}:${OSRM_CAR_PORT}/route/v1/driving/-79.9959,40.4406;-79.9,40.45?overview=false" 60
  wait_http osrm-bike "http://${SERVER}:${OSRM_BIKE_PORT}/route/v1/driving/-79.9959,40.4406;-79.9,40.45?overview=false" 60
  wait_http osrm-foot "http://${SERVER}:${OSRM_FOOT_PORT}/route/v1/driving/-79.9959,40.4406;-79.9,40.45?overview=false" 60
  wait_http map-frontend "http://${SERVER}:13000" 60
}

main() {
  require_data_root
  if [[ "${1:-}" == "--print-plan" ]]; then
    print_plan
    return
  fi

  guard_resources
  download_archives
  extract_archives
  start_backends
  verify_all
  patch_frontend
  wait_http map-frontend "http://${SERVER}:13000" 60
  log "Map backend and frontend are ready"
}

main "$@"
