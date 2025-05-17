#!/usr/bin/env bash
set -euo pipefail

# Папка с вашим проектом
PROJECT_DIR="diploma_pdf"

# Перейти в каталог проекта
cd "$PROJECT_DIR"

# 1) Компиляция LaTeX (два прохода для обновления ссылок/оглавления)
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null

# 2) Удаление всех вспомогательных файлов (включая дочерние папки)
find . -type f \( \
    -name '*.aux' -o \
    -name '*.log' -o \
    -name '*.out' -o \
    -name '*.toc' -o \
    -name '*.lof' -o \
    -name '*.lot' -o \
    -name '*.fls' -o \
    -name '*.fdb_latexmk' -o \
    -name '*.synctex.gz' \
\) -delete

# 3) «Очистка» PDF с помощью qpdf
#    — перепакуем потоки, уберём мусор и перепишем main.pdf
qpdf --stream-data=uncompress main.pdf tmp.pdf
qpdf --stream-data=compress tmp.pdf main.pdf
rm -f tmp.pdf

echo "✅ Сборка завершена, чистый main.pdf готов."
