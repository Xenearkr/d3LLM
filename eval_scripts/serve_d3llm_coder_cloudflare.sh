#!/usr/bin/env bash
# 在服务器上将 SGLang(30000) 经 Cloudflare Quick Tunnel 暴露为公网 HTTPS，供 Cursor BYOK 使用。
#
# 无需 Cloudflare 账号，免费，且支持经 HTTP 代理出网（本服务器 ngrok 免费版不支持代理）。
#
# 用法：
#   bash eval_scripts/serve_d3llm_coder_cloudflare.sh

set -eo pipefail

_script_path="${BASH_SOURCE[0]}"
[[ "${_script_path}" != /* ]] && _script_path="$(pwd)/${_script_path}"
SCRIPT_DIR="$(cd "$(dirname "${_script_path}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
unset _script_path

CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-${HOME}/bin/cloudflared}"
LOCAL_PORT="${LOCAL_PORT:-30000}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/models/cloudflare_d3llm_coder.log}"
PID_FILE="${PID_FILE:-${REPO_ROOT}/models/cloudflare_d3llm_coder.pid}"
URL_FILE="${URL_FILE:-${REPO_ROOT}/models/cloudflare_d3llm_coder.url}"

# 服务器出网代理（ubuntu-lab 可用）
export https_proxy="${https_proxy:-http://ubuntu-lab.zinyy.tech:20172}"
export http_proxy="${http_proxy:-http://ubuntu-lab.zinyy.tech:20172}"
unset all_proxy ALL_PROXY

if [[ ! -x "${CLOUDFLARED_BIN}" ]]; then
  echo "[cloudflare] 未找到 cloudflared: ${CLOUDFLARED_BIN}" >&2
  echo "[cloudflare] 安装: export https_proxy=http://ubuntu-lab.zinyy.tech:20172 && curl -fsSL -o ~/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x ~/bin/cloudflared" >&2
  exit 1
fi

if ! curl -sS --noproxy '*' --connect-timeout 2 "http://127.0.0.1:${LOCAL_PORT}/v1/models" \
  -H "Authorization: Bearer sk-d3llm-local" >/dev/null 2>&1; then
  echo "[cloudflare] 警告: SGLang 未在 127.0.0.1:${LOCAL_PORT} 响应，请先启动 serve_d3llm_coder_sglang.sh" >&2
fi

mkdir -p "$(dirname "${LOG_FILE}")"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "[cloudflare] 已在运行 pid=$(cat "${PID_FILE}")"
  if [[ -f "${URL_FILE}" ]]; then
    url="$(tr -d ' \n\r' < "${URL_FILE}")"
    echo ""
    echo "========== Cursor 配置 =========="
    echo "Override OpenAI Base URL: ${url}/v1"
    echo "OpenAI API Key:           sk-d3llm-local"
    echo "Chat 模型名:              d3llm-dream-coder"
    echo "================================="
    exit 0
  fi
else
  : > "${LOG_FILE}"
  nohup "${CLOUDFLARED_BIN}" tunnel --no-autoupdate --url "http://127.0.0.1:${LOCAL_PORT}" \
    >> "${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"
  echo "[cloudflare] 已启动 pid=$(cat "${PID_FILE}")，日志: ${LOG_FILE}"
fi

echo "[cloudflare] 等待公网 URL..."
for _ in $(seq 1 60); do
  if grep -q "failed to request quick Tunnel" "${LOG_FILE}" 2>/dev/null; then
    echo "[cloudflare] 失败: cloudflared 无法经代理访问 api.trycloudflare.com（本机直连也被防火墙阻断）。" >&2
    echo "[cloudflare] 服务器侧 Quick Tunnel 不可用，请改用 Windows 本机 ngrok + SSH 隧道。" >&2
    exit 1
  fi
  url="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "${LOG_FILE}" 2>/dev/null | grep -v 'api\.trycloudflare\.com' | head -1 || true)"
  if [[ -n "${url}" ]]; then
    echo "${url}" > "${URL_FILE}"
    echo ""
    echo "========== Cursor 配置 =========="
    echo "Override OpenAI Base URL: ${url}/v1"
    echo "OpenAI API Key:           sk-d3llm-local"
    echo "Chat 模型名:              d3llm-dream-coder"
    echo "================================="
    exit 0
  fi
  if ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "[cloudflare] 进程已退出，最近日志:" >&2
    tail -20 "${LOG_FILE}" >&2
    exit 1
  fi
  sleep 2
done

echo "[cloudflare] 暂未获取到 URL，请查看日志: tail -f ${LOG_FILE}" >&2
exit 1
