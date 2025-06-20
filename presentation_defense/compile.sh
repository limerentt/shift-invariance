 #!/bin/bash

# Скрипт для компиляции презентации защиты диплома

echo "🚀 Начинаем компиляцию презентации..."

# Проверяем наличие файла презентации
if [ ! -f "presentation.tex" ]; then
    echo "❌ Ошибка: файл presentation.tex не найден!"
    exit 1
fi

# Компилируем презентацию (два прохода для правильных ссылок)
echo "📝 Первый проход компиляции..."
pdflatex -interaction=nonstopmode presentation.tex

echo "📝 Второй проход компиляции..."
pdflatex -interaction=nonstopmode presentation.tex

# Проверяем успешность компиляции
if [ -f "presentation.pdf" ]; then
    echo "✅ Презентация успешно скомпилирована!"
    echo "📄 Размер файла: $(du -h presentation.pdf | cut -f1)"
    echo "📊 Количество слайдов: $(pdfinfo presentation.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}' || echo "неизвестно")"
    
    # Очищаем временные файлы
    echo "🧹 Очищаем временные файлы..."
    rm -f *.aux *.log *.out *.toc *.nav *.snm *.fls *.fdb_latexmk *.synctex.gz
    
    echo "🎉 Готово! Файл presentation.pdf создан."
    
    # Пытаемся открыть презентацию
    if command -v open > /dev/null; then
        echo "👀 Открываем презентацию..."
        open presentation.pdf
    elif command -v xdg-open > /dev/null; then
        echo "👀 Открываем презентацию..."
        xdg-open presentation.pdf
    else
        echo "💡 Откройте файл presentation.pdf вручную"
    fi
else
    echo "❌ Ошибка компиляции! Проверьте файл presentation.log для деталей."
    exit 1
fi