# A5. Извлечение содержимого вложений

`botkit.extraction` реализует контракт из методички: вложение → текст,
структура или транскрипт. Это не проверка RACI, не ответ на вопросы документа
и не RAG-индексация. Извлечённый результат передаётся соответствующему навыку.
Старые пять позиционных полей `ExtractedContent` сохранены; добавлено `metadata`.
Протокол сервиса по-прежнему находится в `extraction.base`, конкретная реализация
экспортируется из `botkit.extraction`.

## Подключение к боту

```python
from botkit.extraction import ContentExtractionService, ExtractionConfig
from botkit.llm import load_llm
from botkit.usage import PriceBook, SQLiteUsageTracker, usage_context

tracker = SQLiteUsageTracker("data/usage.sqlite3", prices=PriceBook("config/prices.json"))
llm = load_llm(tracker=tracker)
extraction = ContentExtractionService(
    llm, tracker=tracker, config=ExtractionConfig(pdf_mode="auto"),
)

async def handle_document(message):
    with usage_context(skill_context="DOCUMENT_REVIEW"):
        content = await extraction.extract(message.attachments[0])
    # content.text / structured_data / warnings / source_confidence / metadata
    return content

# llm.aclose() вызывается владельцем при остановке приложения.
```

Telegram и VK уже отдают нужный `Attachment` с `read_bytes()`. Ни SDK мессенджера,
ни отдельный HTTP-клиент распознавания в навыке не нужны. Без мессенджера:

```python
content = await extraction.extract_bytes(
    document_bytes, "application/octet-stream", filename="document.pdf",
)
```

MIME уточняется по сигнатуре файла; для текстовых файлов с generic MIME
используется расширение. При уточнении добавляется warning. Имя файла не
используется как путь для записи, команды оболочки не запускаются.

`python -m botkit run --platform telegram vk` теперь подключает A5 к
демонстрационному боту: он обрабатывает и сообщение только с вложением,
передаёт содержимое и предупреждения в контекст ответа. До 5 вложений и
60 000 символов сериализованного контекста на сообщение; превышение даёт
ошибку, а не скрытую обрезку. Это пример маршрутизации, не новая реализация
всех навыков старых ботов. По умолчанию документ не сохраняется на диск.

## PDF: выбор режима

| Режим | Поведение | Сетевые обращения |
|---|---|---|
| `text` | Извлечь текстовый слой и доступные таблицы; отметить пустые/подозрительные страницы | Нет |
| `vision` | Преобразовать каждую страницу в PNG и распознать моделью | По одному логическому vision-вызову на страницу |
| `auto` | Использовать текстовый слой; сканы, подозрительный текст и страницы с крупными изображениями распознать через vision | Только для выбранных страниц |

`auto` — эвристика, не доказательство правильности OCR. Страница выбирается
для vision, если текст пустой, содержит признаки повреждённой кодировки,
мало букв/цифр (менее 30) либо изображения занимают не менее 20% площади.
Короткая титульная страница тоже может попасть в vision. Диаграммы, составленные
только из векторных линий, эта эвристика может не определить — для документов,
где важен внешний вид, используйте `vision` явно.

Без LLM режим `auto` возвращает доступный текст и `vision_unavailable` на
нуждающихся в OCR страницах. `vision` без LLM — явная ошибка. Сбой OCR не
превращается в молчаливый успех с половиной документа. Если нативный парсер
не смог прочитать PDF, `auto` с LLM пробует независимый путь отрисовки;
лимиты при этом не обходятся. Запароленные PDF без пароля отклоняются.

Сохраняются порядок и номера страниц:

```python
content.structured_data["pages"][0]
# {
#   "page_number": 1,
#   "text": "...",
#   "tables": [{"rows": [["Имя", "Роль"], ["...", "..."]]}],
#   "method": "text" или "vision",
#   "warnings": [...],
#   "metadata": {...}  # сведения о vision-модели, если применялась
# }
```

Преобразование PDF в картинки доступно независимо от извлечения и LLM:

```python
from botkit.extraction import render_pdf

pages = await render_pdf(pdf_bytes)
for page in pages:
    # page.page_number, page.data (PNG bytes), page.width, page.height
    image_result = await llm.ainvoke_with_image(
        "Извлеки таблицу в JSON", page.data, mime_type=page.mime_type,
    )
```

Последний пример намеренно делает собственные модельные вызовы; не вызывайте
его дополнительно к сервису с `ExtractionConfig(pdf_mode="vision")`: это
удвоит обращения. Режим задаётся через конфигурацию, не аргументом `extract`.
Страницы отрисовываются PDFium с ограничением числа пикселей; бинарники идут
в Python-зависимости, отдельная установка Poppler на сервер бота не нужна.
Все обращения к PDFium внутри A5 защищены общей блокировкой: библиотека не
допускает параллельные вызовы из разных потоков. См. [официальные ограничения PDFium](https://pypdfium2.readthedocs.io/en/stable/python_api.html).

## Форматы и ограничения извлечения

| Формат | Что возвращается |
|---|---|
| PDF | Текст и таблицы по страницам, OCR при необходимости |
| DOCX | Параграфы и таблицы основного тела в порядке документа, включая вложенные таблицы |
| TXT, MD, CSV, TSV | Текст; разделители и строки сохраняются |
| JSON | Исходный текст и разобранное значение; повреждённый JSON сохраняется с warning |
| HTML | Видимый текст без script/style; внешние ресурсы не загружаются |
| XML | Текст без выполнения/разрешения XML-сущностей |
| PNG, JPEG, WebP, BMP, TIFF, GIF | Vision JSON: `text`, `tables`, `warnings`; для многостраничных/анимированных файлов только первый кадр с предупреждением |
| XLSX (extra `office`) | Листы и ячейки; формулы сохраняются как текст, не исполняются |
| PPTX (extra `office`) | Текст и таблицы по слайдам, текст сгруппированных фигур |
| Аудио | Только при отдельно переданном ASR-провайдере, см. ниже |

Для XLSX/PPTX: `pip install -e ".[office]"` или `uv sync --extra office`.
DOCX не извлекает колонтитулы/сноски, изображения внутри Office-документов не
распознаются автоматически. PPTX не отрисовывает диаграммы и не читает заметки
докладчика; порядок фигур не всегда равен визуальному порядку. Эти ограничения
возвращаются в `warnings`. Для полного визуального разбора экспортируйте Office
документ в PDF и используйте vision. XLSX читается в [read-only режиме](https://openpyxl.readthedocs.io/en/stable/optimized.html).

Старые бинарные DOC/XLS/PPT, RTF, ODT, архивы ZIP, macro-enabled документы
и неизвестные форматы не конвертируются скрытым запуском LibreOffice:
возвращается `UnsupportedContentError` с просьбой конвертировать файл.

UTF-8/BOM, UTF-16/BOM, UTF-32/BOM распознаются напрямую; fallback на CP1251
всегда помечается предупреждением о необходимости проверить кодировку.
Картинки нормализуются в PNG: ориентация EXIF учитывается, прозрачность
компонуется на белом фоне, слишком большое изображение уменьшается с warning.

## Vision и внутренний Qwen

Обращения проходят только через A1: `LLMProvider.ainvoke_with_image_b64`.
Изображение встроено в запрос как `data:image/png;base64,...`; модели не
нужен доступ к URL мессенджера. Ожидается JSON с `text`, `tables[].rows`,
`warnings`; неверная схема или обрезанный ответ вызывают `InvalidLLMResponse`
и могут запустить настроенную A1 fallback-цепочку. Доменный RACI JSON при
необходимости извлекайте своим навыком, а не меняйте универсальную OCR-схему.

По уточнению владельца сервера **2 секунды — это пауза между запросами**,
не таймаут генерации. Рекомендуемые параметры `.env`:

```dotenv
LLM_PROVIDER=hub
LOCAL_HUB_API_BASE=<выданный адрес с /ai/v1>
LOCAL_HUB_API_KEY=<ключ>
LOCAL_HUB_MODEL_NAME=qwen3.8-27b-w4a16-awq
LOCAL_HUB_REQUEST_INTERVAL=2
LOCAL_HUB_STREAMING=false
LLM_TIMEOUT=180
EXTRACTION_VISION_TIMEOUT=180
```

В обычном запросе явно передаётся `stream=false`. Попытка вызвать `astream`
при отключённом streaming отклоняется до сети. Очередь A1 общая для моделей
на одном endpoint в одном event loop: запросы идут последовательно, после
завершения предыдущего выдерживается минимум 2 секунды, включая переключение
на другую модель этого же хаба. Пауза и очередь входят в общий дедлайн.
Это не распределённый limiter: при нескольких процессах/репликах нужна общая
очередь на стороне развёртывания. Остальные провайдеры по умолчанию без этой паузы.

При включённом fallback `LLM_TIMEOUT` — бюджет всей цепочки; A1 резервирует
его части для запасных моделей. Если Qwen нужен весь таймаут 180 секунд,
используйте `load_llm("hub", fallback=False)` или увеличьте общий бюджет и
`LLM_PRIMARY_TIMEOUT` осознанно. Лимит A5 `vision_timeout` применяется к одному
логическому вызову страницы, а не ко всему многостраничному PDF.

Проверка сертификата по умолчанию включена. Присланный владельцем сервера
пример с `verify=False` воспроизводится явным `LOCAL_HUB_VERIFY_SSL=false`;
эта настройка ослабляет TLS-защиту. Предпочтительно указать доверенный
`LOCAL_HUB_CA_BUNDLE`. Отключение проверки не исправило текущий обрыв TLS.
Ключи и адреса из старого `.env` не копируются в библиотеку.

## Аудио — отдельная возможность

Наличие vision-модели не означает наличие распознавания речи. Зарегистрируйте
отдельный ASR. Для предоставленного сервера Pneuma уже есть готовый клиент:

```python
from botkit.llm import load_transcriber

asr = load_transcriber(tracker=tracker)  # ASR_PROVIDER=pneuma, PNEUMA_BASE_URL в env
extraction = ContentExtractionService(llm, transcriber=asr, tracker=tracker)
# При завершении приложения: await asr.aclose(), если asr is not None.
```

Он загружает multipart `audio`, ждёт асинхронную задачу и возвращает транскрипт
с таймкодами/спикерами сервера без переформулирования. Поддержаны прямой API и
веб-прокси. Настройка, ограничения и отдельный тестовый скрипт — в
[Pneuma ASR](Pneuma%20ASR.md). `python -m botkit run` подключает его из env автоматически.

Для **другого** ASR gateway, действительно поддерживающего OpenAI
`/audio/transcriptions`, остаётся прежний вариант:

```python
from botkit.llm import LLMConfig, LLMGateway

asr = LLMGateway(LLMConfig(
    "openai_compatible", "your-asr-model", "https://asr.example/v1", "your-key",
), tracker=tracker)
extraction = ContentExtractionService(llm, transcriber=asr, tracker=tracker)
```

Поддерживаемые MIME-типы: WAV, MP3, OGG/Opus, M4A, FLAC, WebM, AAC, WMA,
если их принимает выбранный ASR. MP3 и другие аудиофайлы с MIME
`application/octet-stream` определяются по расширению; OGG/WAV/FLAC/MP3 с ID3
также по сигнатуре. `.opus` направляется как OGG. Голосовые Telegram/VK уже
нормализуются адаптерами в аудиовложения. Отдельный таймаут аудио:
`EXTRACTION_AUDIO_TIMEOUT=600`, не `EXTRACTION_VISION_TIMEOUT`.
Через `load_transcriber()` можно включить resumable-загрузку Pneuma и доработку
транскрипта локальной LLM: `PNEUMA_UPLOAD_PROTOCOL=chunked`,
`ASR_POSTPROCESS_ENABLED=true`. [Подробности и API](Pneuma%20ASR.md#новый-протокол-и-доработка-llm).
A5 использует `LLMResponse.response` (обработанный текст), не `raw` (исходный ASR).
В metadata добавляются `postprocessed`, `postprocess_reason`, `postprocessor_model`;
в warnings — `audio_transcript_postprocessed` или `audio_transcript_postprocess_skipped`.
Общий `EXTRACTION_AUDIO_TIMEOUT` включает ASR и доработку; отдельные дедлайны
`PNEUMA_TIMEOUT` и `ASR_POSTPROCESS_TIMEOUT` также действуют.

Транскрипт имеет `kind=transcript`, confidence `low`. Без ASR аудио даёт
понятную ошибку; автоматически отправлять его в Qwen нельзя. Закрытие `asr`
выполняет владелец. Этот путь проверен протокольными тестами, не реальной ASR-моделью.

## Лимиты, предупреждения, журнал

По умолчанию: файл 20 MiB, 30 страниц/слайдов/листов, максимум 10 vision-страниц,
200 000 символов текста, 50 000 ячеек, 25 млн входных пикселей, 4 млн пикселей
отрисованной страницы, 50 MiB суммарных PNG при `render_pdf`. OOXML: до 2000
элементов и 100 MiB распакованного содержимого; XML entities и макросы отклоняются.
Лимит vision-страниц относится к логическим страницам; retries/fallback могут
увеличить реальное число и стоимость запросов. Большой PDF разделите явно.

Лимиты меняются через `ExtractionConfig`; основные — через переменные
`EXTRACTION_*` в `.env.example`. Извлечение и отрисовка выполняются вне event
loop. `asyncio.to_thread` не умеет принудительно остановить уже работающий
парсер: для публичного высоконагруженного сервиса дополнительно изолируйте
парсеры отдельными worker-процессами с OS-лимитами CPU/RAM. Эти проверки не
являются полноценной песочницей для произвольных враждебных документов.

OCR/ASR всегда имеют confidence не выше `low`, даже при валидном JSON.
Предупреждения модели, парсера, выбранного режима и пропущенных частей доступны
вызывающему коду. Это нужно учитывать в навыке, а не скрывать от пользователя.
У PDF предупреждения помечены номером страницы.

A2 записывает `kind=extraction`, размер, SHA-256, тип, длительность, статус,
число предупреждений/страниц и confidence. Vision/ASR отдельно учитываются
как LLM-вызовы. Сохраняется внешний user/chat/skill-контекст; для страниц есть
`extraction_page`. Содержимое и модельные warnings не пишутся в extraction
metadata. Запись исходных LLM-текстов по-прежнему требует явного `LOG_TEXT=true`.
Хеши и длины сами по себе не делают персональные документы анонимными.

## Быстрая проверка

```powershell
# Только локальное извлечение:
uv run --no-sync python examples/extract_attachment.py document.pdf --pdf-mode text
# Включить модель и сохранить результат в НОВЫЙ файл:
uv run --no-sync python examples/extract_attachment.py scan.pdf --pdf-mode auto `
  --provider hub --env .env --output result.json
```

Первый пример не делает LLM-запросов. Второй может быть платным, записанный JSON
может содержать персональные данные. Существующий output-файл не перезаписывается.
Для нового формата реализуйте `Extractor` и вызовите `service.register(extractor)`;
последний зарегистрированный парсер проверяется первым.
