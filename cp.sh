#!/usr/bin/env bash
set -euo pipefail

# cp.sh — commit & push
# Использование: ./cp.sh [-d] [-p] ["сообщение"]
#   -d  mirror: включает удаления (локальное удаление -> удаление в remote)
#       без -d удаления игнорируются (безопасный режим)
#   -p  preview: показать что будет закоммичено, без коммита/пуша
# Без сообщения используется дата.

cd "$(dirname "$0")"

DELETE=false
PREVIEW=false
MSG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d) DELETE=true; shift ;;
    -p) PREVIEW=true; shift ;;
    -h) echo "Использование: $0 [-d] [-p] [\"сообщение коммита\"]"; echo "  -d  удалить в remote файлы удаленные локально"; echo "  -p  preview без коммита"; exit 0 ;;
    --) shift; break ;;
    -*) echo "Неизвестный ключ: $1 (допустимы -d, -p)" >&2; exit 1 ;;
    *) MSG="$1"; shift; break ;;
  esac
done

if [[ $# -gt 0 ]]; then
  echo "Лишний аргумент: $*" >&2
  exit 1
fi

if [[ -z "$MSG" ]]; then
  MSG="Update: $(date '+%Y-%m-%d %H:%M:%S')"
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Ошибка: не git-репозиторий." >&2
  exit 1
fi

if [[ -z "$(git status --porcelain)" ]]; then
  echo "Нет изменений для коммита."
  exit 0
fi

if "$DELETE"; then
  git add -A
else
  git add -A
  # снять с индекса удаления — безопасный режим
  mapfile -t deleted < <(git diff --cached --name-only --diff-filter=D || true)
  if [[ ${#deleted[@]} -gt 0 && -n "${deleted[0]}" ]]; then
    printf 'Пропущены удаления (используй -d чтобы удалить в remote):\n' >&2
    printf '  D %s\n' "${deleted[@]}" >&2
    git restore --staged -- "${deleted[@]}" 2>/dev/null || git reset -q -- "${deleted[@]}"
  fi
fi

if git diff --cached --quiet; then
  echo "Нет staged изменений для коммита."
  exit 0
fi

if "$PREVIEW"; then
  echo "Preview (-p) ${DELETE:+с -d (с удалениями):-без -d (без удалений)}:"
  git diff --cached --stat
  echo "---"
  git diff --cached --name-status
  # откатить staged чтобы не менять индекс при preview
  git reset -q
  exit 0
fi

git commit -m "$MSG"
git push

echo "Коммит и пуш выполнены: $MSG"
if "$DELETE"; then
  echo "Режим: mirror (-d) — удаления синхронизированы."
else
  echo "Режим: safe — удаления пропущены (используй -d для удаления в remote)."
fi
