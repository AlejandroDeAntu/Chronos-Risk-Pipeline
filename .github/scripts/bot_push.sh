#!/usr/bin/env bash
# Empuja la rama actual con la llave de despliegue del bot.
#
# La llave solo existe en disco durante este paso: checkout corre con
# persist-credentials: false, así que ni `pip install` ni el pipeline (código
# de terceros) pueden leerla. Es la única identidad exenta de la regla que
# exige CI en main.
set -euo pipefail

if [ -z "${BOT_DEPLOY_KEY:-}" ]; then
  echo "Falta el secreto BOT_DEPLOY_KEY: no se puede empujar a main." >&2
  exit 1
fi

key_dir="$(mktemp -d)"
trap 'rm -rf "$key_dir"' EXIT
printf '%s\n' "$BOT_DEPLOY_KEY" > "$key_dir/key"
chmod 600 "$key_dir/key"
# Llave pública de github.com fijada (https://docs.github.com/authentication/
# keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints).
printf '%s\n' \
  "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl" \
  > "$key_dir/known_hosts"

GIT_SSH_COMMAND="ssh -i $key_dir/key -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$key_dir/known_hosts" \
  git push "git@github.com:${GITHUB_REPOSITORY}.git" "HEAD:${GITHUB_REF_NAME}"
