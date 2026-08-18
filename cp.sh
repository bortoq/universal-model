#!/usr/bin/env bash
set -euo pipefail

# cp.sh — commit & push текущего проекта
# Использование: ./cp.sh ["сообщение коммита"]
# Без аргумента используется сообщение по умолчанию.

cd "$(dirname "$0")"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Ошибка: не git-репозиторий." >&2
    exit 1
fi

# Есть ли изменения для коммита (tracked + untracked)?
if [ -z "$(git status --porcelain)" ]; then
    echo "Нет изменений для коммита."
    exit 0
fi

MSG="${1:-Update: $(date '+%Y-%m-%d %H:%M:%S')}"

git add -A
git commit -m "$MSG"
git push

echo "Коммит и пуш выполнены: $MSG"
