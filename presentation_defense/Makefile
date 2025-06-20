# Makefile для компиляции презентации
MAIN = presentation
TEX = $(MAIN).tex
PDF = $(MAIN).pdf

# Основная цель
all: $(PDF)

# Компиляция PDF
$(PDF): $(TEX)
	@echo "Компиляция презентации..."
	@pdflatex -interaction=nonstopmode $(TEX) > /dev/null 2>&1
	@pdflatex -interaction=nonstopmode $(TEX) > /dev/null 2>&1
	@echo "Презентация скомпилирована: $(PDF)"

# Очистка временных файлов
clean:
	@echo "Очистка временных файлов..."
	@rm -f *.aux *.log *.out *.toc *.nav *.snm *.fls *.fdb_latexmk *.synctex.gz
	@echo "Временные файлы удалены"

# Полная очистка (включая PDF)
distclean: clean
	@rm -f $(PDF)
	@echo "PDF файл удален"

# Быстрая компиляция (один проход)
quick:
	@echo "Быстрая компиляция..."
	@pdflatex -interaction=nonstopmode $(TEX) > /dev/null 2>&1
	@echo "Быстрая компиляция завершена"

# Открыть PDF после компиляции
view: $(PDF)
	@echo "Открытие презентации..."
	@open $(PDF) 2>/dev/null || xdg-open $(PDF) 2>/dev/null || echo "Не удалось открыть PDF автоматически"

# Справка
help:
	@echo "Доступные команды:"
	@echo "  make all     - Полная компиляция презентации"
	@echo "  make quick   - Быстрая компиляция (один проход)"
	@echo "  make clean   - Удалить временные файлы"
	@echo "  make distclean - Удалить все сгенерированные файлы"
	@echo "  make view    - Скомпилировать и открыть презентацию"
	@echo "  make help    - Показать эту справку"

.PHONY: all clean distclean quick view help 