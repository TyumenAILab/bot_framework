# Голосовые сообщения и MP3 — Pneuma ASR

Фреймворк поддерживает **два различных HTTP-протокола Pneuma** и необязательную
доработку транскрипта через существующий A1 LLM Gateway. Компоненты находятся в
`botkit.llm`, не требуют SocratiQ, SQLAlchemy или кода конкретного мессенджера.
Старый протокол и отсутствие доработки остаются значениями по умолчанию.

## Новый протокол и доработка LLM

Реализация `ChunkedPneumaTranscriber` перенесена из чат-бота и соответствует
предоставленным `audio_processing_client.py` и `resumable_chunked_upload_example.ipynb`:

1. `POST /uploads` — filename, size, user_id, device (JSON, при 415/422 — form).
2. `PATCH /uploads/{upload_id}` — двоичные чанки по 16 MiB, `Upload-Offset`.
3. `GET /uploads/{upload_id}` — сверка `received_bytes` после потерянного ответа.
4. `POST /uploads/{upload_id}/complete` — запуск распознавания. При обрыве ответа
   сначала проверяется существующий job, а не безусловно создаётся повторный.
5. `GET /transcriptions/{job_id}` и `/result` — ожидание и получение текста.
6. При включённой доработке **только текст**, не аудио, передаётся локальной LLM.

Поддерживаются строковый transcript, объект `text`/`transcript`/`segments` и
списки текстовых сегментов. Неизвестные объекты не превращаются в пользовательский
текст через `str()`/JSON. По умолчанию новый клиент принимает до 20 MiB аудио
(`MAX_ATTACHMENT_BYTES`), 2 MiB ответа HTTP и 20 000 символов транскрипта.
Лимит файла можно переопределить в `UploadConfig`; чанки по 16 MiB применяются
и к файлам больше одного чанка. В чат-боте сохраняется более строгий предел 10 MiB.

Пример `.env` для **нового сервера** (адреса и ключи заполните самостоятельно):

```dotenv
ASR_PROVIDER=pneuma
PNEUMA_UPLOAD_PROTOCOL=chunked
PNEUMA_BASE_URL=https://pneuma.example
PNEUMA_API_KEY=
PNEUMA_USER_ID=
PNEUMA_DEVICE=cuda
PNEUMA_TIMEOUT=600
PNEUMA_CONNECT_TIMEOUT=15
PNEUMA_REQUEST_TIMEOUT=30
PNEUMA_POLL_INTERVAL=5
PNEUMA_UPLOAD_STATE_DB=data/pneuma_uploads.sqlite3
PNEUMA_VERIFY_SSL=true

ASR_POSTPROCESS_ENABLED=true
ASR_POSTPROCESS_LLM_PROVIDER=hub
ASR_POSTPROCESS_TIMEOUT=30
ASR_POSTPROCESS_MAX_CHARS=20000
ASR_POSTPROCESS_MAX_TOKENS=8192
ASR_POSTPROCESS_ON_ERROR=raise
LOCAL_HUB_API_BASE=https://your-local-llm.example/ai/v1
LOCAL_HUB_API_KEY=
LOCAL_HUB_MODEL_NAME=qwen3.8-27b-w4a16-awq
LOCAL_HUB_REQUEST_INTERVAL=2
LOCAL_HUB_STREAMING=false
EXTRACTION_AUDIO_TIMEOUT=660
LOG_TEXT=false
```

`ASR_POSTPROCESS_LLM_PROVIDER` принимает `hub`, `vllm`, `openai_compatible`:
используются обычные `LOCAL_HUB_*` или `VLLM_*` настройки фреймворка.
`ASR_POSTPROCESS_MODEL` необязательно переопределяет имя модели. Адрес локального
сервера задаёт оператор: название провайдера само по себе не проверяет, где
физически размещён endpoint. Фабрика создаёт отдельный клиент с `fallback=False`,
`max_retries=0`, `supports_streaming=False`; глобальные fallback в 302.ai/другую
модель не применяются. При использовании vLLM задайте `LLM_REQUEST_INTERVAL=2`,
если сервер требует паузу; для hub она уже составляет 2 секунды по умолчанию.

**Без LLM:** `ASR_POSTPROCESS_ENABLED=false`. Тогда параметры модели и её ключ
для ASR не нужны. **Старый сервер:** `PNEUMA_UPLOAD_PROTOCOL=legacy`; доработка
также совместима с ним. `PNEUMA_API_MODE=proxy` и `PNEUMA_CANCEL_ON_TIMEOUT`
относятся только к legacy; новый протокол не имитирует неподтверждённый `/cancel`.

### Вызов из любого бота

```python
from dotenv import load_dotenv
from botkit.llm import load_transcriber
from botkit.usage import usage_context

async def recognize(audio_bytes, user_ref, message_ref, tracker=None):
    load_dotenv()
    asr = load_transcriber(tracker=tracker)
    if asr is None:
        raise ValueError("ASR_PROVIDER=pneuma is required")
    async with asr:
        with usage_context(user_id=user_ref, request_id=message_ref):
            result = await asr.atranscribe(
                audio_bytes, mime_type="audio/mpeg", filename="voice.mp3", timeout=660,
            )
            return result.response, result.raw, result.meta
```

В долгоживущем боте создайте один экземпляр при старте и закройте `aclose()`
при остановке; A5 и runtime уже используют эту фабрику.
`result.response` — текст после доработки, `result.raw` — исходный текст Pneuma.
В `meta`: job_id, poll_count, upload_protocol, postprocessed, postprocess_reason,
postprocessor_model. Token usage ASR остаётся неизвестным; токены доработки
учитываются отдельным настоящим LLM-вызовом в A2, не подменяют статистику ASR.

`PNEUMA_USER_ID` имеет приоритет над `usage_context(user_id=...)`. Передавайте
псевдоним, а не имя человека; фреймворк **не анонимизирует** этот идентификатор сам.
Без обоих значений новый клиент явно откажется отправлять файл. `request_id`
должен быть одинаковым для повтора **того же** сообщения и отличаться для нового.
Без request_id одинаковое аудио одного пользователя может использовать один job
до истечения TTL. Передача общего PNEUMA_USER_ID требует уникальных request_id
между пользователями. Демонстрационный smoke-скрипт задаёт синтетические ID.

### Возобновление и хранение

`MemoryUploadStore` по умолчанию хранит до 1000 состояний только в RAM.
`PNEUMA_UPLOAD_STATE_DB` включает `SQLiteUploadStore`, переживающий перезапуск.
Можно передать собственный `load_transcriber(store=...)` с async-методами
`get(key)`, `put(key, value)`, `prune()` (протокол `UploadStore`).
Сохраняются лишь SHA256-ключ, upload/job ID, смещение и время; **не** аудио,
транскрипт, исходное имя файла или API-ключ. TTL — 24 часа, очистка при обращении
к хранилищу. Для восстановления после перезапуска вызывающее приложение должно
повторно предоставить те же байты аудио и контекст; это не фоновая очередь.

Один экземпляр клиента сериализует одинаковые загрузки. SQLite обеспечивает
целостность метаданных, но не распределённую блокировку загрузки: несколько
worker-процессов не должны одновременно обрабатывать одно сообщение.
Потеря ответа самого первого `POST /uploads` не позволяет восстановить неизвестный
ID; этот запрос не повторяется автоматически. При timeout/отмене известные ID
сохраняются, сервер может продолжать задачу. Ключи состояния включают адрес,
пользователя, устройство, расширение, request_id и SHA256 аудио.

### Контроль доработки и расширение

`LLMTranscriptProcessor.aprocess(text)` возвращает `TranscriptResult(text, processed, reason)`.
Перед вызовом удаляются только известные префиксы таймкодов/спикеров Pneuma
(можно отключить `strip_headers=False`). По умолчанию LLM восстанавливает только
пунктуацию/регистр. Детерминированная проверка `validate_transcript()` сравнивает
слова, их порядок и числовые литералы со знаками; изменение возвращает исходный
текст с `processed=False, reason=content_changed`. Это не гарантия сохранения всех
оттенков смысла: пользовательское подтверждение ASR по-прежнему полезно.

Ошибка модели/JSON по умолчанию прерывает операцию; при явно выбранном
`ASR_POSTPROCESS_ON_ERROR=original` возвращается исходный нормализованный текст
с `reason=llm_error`. Отмена задачи не подавляется ни в одном режиме. Пустой ASR
не отправляется модели. A5 всегда оставляет `audio_transcript_unverified` и
добавляет предупреждение о применённой/пропущенной доработке.

`TranscriptionPipeline(asr, processor)` совместим с `AudioTranscriptionProvider`.
Можно передать собственный `TranscriptProcessor` для другой логики доработки;
его `aprocess()` должен вернуть `TranscriptResult`. Переданные компоненты не
закрываются автоматически, если не задано `owns_components=True`. Фабрика владеет
созданными клиентами и закрывает их через pipeline.

Для совмещения доработки и доменной классификации в **одном** запросе есть
`LLMTranscriptProcessor.arequest(text, system_prompt=..., validator=..., context=...)`.
Этот метод не применяет стандартный parser: приложение обязано проверить свою
JSON-схему и вызвать `validate_transcript()` для текста. Так работает SocratiQ:
общая реализация загрузки/LLM-запроса/проверки текста — в botkit, категории
нежелательных намерений и статичные ответы — в самом боте. В SocratiQ используется
его `SOCRATIQ_INPUT_LLM_*`, а не второй автоматический postprocess-вызов.

Новый загрузчик и проверка обработки не записывают текст в свою телеметрию.
LLM Gateway подчиняется `LOG_TEXT`; оставьте `false` для приватных транскриптов.
HTTP redirects запрещены, TLS включён. Сервер Pneuma может сохранять аудио и
текст независимо от локального клиента; серверные артефакты не удаляются.

## Старый протокол

`PneumaTranscriber` подключает предоставленный сервер из `pneuma-vm-main`
к интерфейсу `AudioTranscriptionProvider` и A5. Код сервера не копируется в
библиотеку. На машине бота не нужны CUDA, NeMo, T-one, FFmpeg или отдельный
OpenAI SDK: аудио декодируется и распознаётся на сервере.

Клиент реализован по `services/diarized-api/nemo_diarized_transcriber/api.py`
и `services/web-proxy/app/main.py` предоставленной копии. Это **не** OpenAI
`/audio/transcriptions`: поле загрузки называется `audio`, результат асинхронный.

### Настройка в `.env` (legacy)

Добавьте блок из `.env.example` в свой файл, не перезаписывая существующие ключи:

```dotenv
ASR_PROVIDER=pneuma
PNEUMA_UPLOAD_PROTOCOL=legacy
ASR_POSTPROCESS_ENABLED=false
PNEUMA_BASE_URL=https://pneuma.example
PNEUMA_API_MODE=direct
PNEUMA_API_KEY=
PNEUMA_TIMEOUT=600
PNEUMA_REQUEST_TIMEOUT=180
PNEUMA_POLL_INTERVAL=2
PNEUMA_VERIFY_SSL=true
PNEUMA_TRUST_ENV=true
PNEUMA_CANCEL_ON_TIMEOUT=true
EXTRACTION_AUDIO_TIMEOUT=600
```

Замените `https://pneuma.example` на доступный **с машины бота** адрес.
Для прямого API задаётся корень сервера/префикс размещения: библиотека добавит
`/transcriptions`. Не добавляйте `/v1` или `/transcriptions` в базовый адрес.
Для веб-прокси используйте одновременно:

```dotenv
PNEUMA_BASE_URL=https://pneuma.example/api
PNEUMA_API_MODE=proxy
```

Префикс `/api` не добавляется автоматически — это позволяет использовать любое
размещение reverse proxy, например `/pneuma/api`. Приватные адреса исходного
сервера в библиотеку не зашиты. Без `ASR_PROVIDER` или при `disabled` ASR выключен;
неподдерживаемое значение/невалидный адрес вызывают ошибку настройки, а не тихое
переключение на Qwen. Функция `load_transcriber()` сама `.env` не читает: CLI
загружает его автоматически, в своём приложении вызовите `load_dotenv()`.

`PNEUMA_API_KEY` — необязательный `Authorization: Bearer ...` для внешнего
шлюза, **не ключ Qwen**. У прямого API в предоставленном коде нет встроенной
проверки ключа. Authentik/SSO на веб-входе может потребовать отдельной серверной
настройки машинного доступа: клиент не выполняет браузерный вход. Cookie
`pneuma_session` нужен прокси для принадлежности задач и поддерживается клиентом;
он не заменяет внешнюю авторизацию. Один экземпляр безопасно сохраняет этот
контекст при параллельных отправках. Используйте отдельный клиент для отдельных
учётных записей/тенантов, если это требуется вашим приложением.

TLS проверяется по умолчанию. Для своего CA укажите
`PNEUMA_CA_BUNDLE=C:/certs/pneuma-ca.pem`; `PNEUMA_VERIFY_SSL=false` сознательно
ослабляет защиту. `PNEUMA_TRUST_ENV` управляет использованием системного proxy.
Автоматического перехода по HTTP redirects нет — в том числе на страницу SSO.

## Использование в библиотеке

```python
from dotenv import load_dotenv
from botkit.extraction import ContentExtractionService, ExtractionConfig
from botkit.llm import load_transcriber

async def recognize(attachment, tracker=None):
    load_dotenv()
    asr = load_transcriber(tracker=tracker)
    if asr is None:
        raise ValueError("Настройте ASR_PROVIDER=pneuma")
    async with asr:
        extraction = ContentExtractionService(
            transcriber=asr, tracker=tracker, config=ExtractionConfig.from_env(),
        )
        result = await extraction.extract(attachment)
        return result.text
```

В длительно работающем боте создавайте один `asr` при старте, передавайте
`transcriber=asr` в A5 и вызывайте `await asr.aclose()` при остановке.
`ContentExtractionService` не закрывает переданный клиент. Встроенный
`python -m botkit run --platform telegram vk` уже делает это автоматически.
Ему по-прежнему нужна чат-модель: аудио идёт только в Pneuma, затем **транскрипт**
передаётся настроенной LLM для ответа. При включённом fallback текст может
попасть в 302.ai. Для одного лишь распознавания LLM не требуется.

Без A5 также можно вызвать провайдер напрямую:

```python
from botkit.llm import PneumaConfig, PneumaTranscriber

async def recognize_bytes(audio_bytes):
    async with PneumaTranscriber(PneumaConfig("https://pneuma.example")) as asr:
        result = await asr.atranscribe(
            audio_bytes, mime_type="audio/mpeg", filename="recording.mp3",
        )
        return result.raw
```

Возвращается `LLMResponse`: `raw`/`response` содержат исходный транскрипт сервера
с таймкодами и спикерами. `provider=pneuma`, `model=t-one` — метка бекенда
предоставленной реализации, не выбор модели через API. `meta` содержит ID задачи,
число опросов, доступную длительность аудио, размер и хеш. `language` не поддержан
сервером: переданный параметр отклоняется явно. Изменять модель, язык, параметры
диаризации или базу голосов следует на сервере; библиотека не меняет его конфигурацию.

## Голосовые и файлы

- Telegram: `voice`, `audio`, MP3 как `document`.
- VK: `audio_message` (OGG, иначе MP3), доступный `audio.url`, MP3 как `doc`.
- A5 распознаёт MP3, OGG/Opus, WAV, M4A, FLAC, AAC и WMA по MIME/расширению;
  OGG/WAV/FLAC/MP3 с ID3 — также по сигнатуре. `.opus` трактуется как OGG.
- `audio/webm` поддерживается интерфейсом A5; возможность декодирования конкретного
  кодека зависит от установленного серверного окружения. Видео автоматически
  не отправляется на ASR. MP3 без доступного URL в VK скачать невозможно.

Полученный A5 объект имеет `kind=transcript`, `source_confidence=low` и предупреждение
`audio_transcript_unverified`. Пустой текст дополнительно получает `empty_extraction`.
Это результат распознавания, а не гарантия точности текста или определения спикеров.

## Таймауты, ошибки и конфиденциальность

Одна загрузка `POST` → опрос состояния `GET` → чтение `/result`.
`PNEUMA_TIMEOUT` ограничивает всю операцию (загрузку, очередь, распознавание,
опросы, результат), по умолчанию 600 секунд. `PNEUMA_REQUEST_TIMEOUT=180` —
сетевой таймаут отдельного запроса, `PNEUMA_POLL_INTERVAL=2` — пауза между опросами,
**не таймаут распознавания**. При A5 также действует `EXTRACTION_AUDIO_TIMEOUT`;
эффективный общий предел — меньший из двух. Длинные записи требуют больших
обоих значений. Операция не использует streaming генерации.

На timeout/отмену клиента direct-режим по умолчанию делает best-effort
`POST /transcriptions/{job_id}/cancel` только для собственной созданной задачи.
Это может добавить до 5 секунд; отмена сервера не гарантирована. Отключение:
`PNEUMA_CANCEL_ON_TIMEOUT=false`. У предоставленного веб-прокси такого маршрута
нет: после прекращения ожидания работа на сервере может продолжиться.
Если связь оборвалась до получения ID, отменить неизвестную задачу невозможно.

Запрос загрузки автоматически **не повторяется**: повтор может создать вторую
задачу. Ошибки HTTP/сети/JSON/состояния возвращаются явно, в том числе на опросе;
fallback в Qwen/302.ai для аудио не выполняется. Серверные `error`, файловые пути
и URL из ответов не показываются и не используются для последующих запросов.

`MAX_ATTACHMENT_BYTES` ограничивает скачивание/передачу (по умолчанию 20 MiB;
в env стенда 5 MiB). A5 ограничивает текст `EXTRACTION_MAX_TEXT_CHARS` (200 000).
Ответ API ограничен 2 MiB (`PneumaConfig.max_response_bytes`). Локальные пути и
оригинальные имена файлов не передаются: загрузка называется `audio.<расширение>`.
Пользовательские файлы, transcript и speaker database сервер может сохранять:
библиотека **не удаляет** его артефакты и не вызывает `/cleanup`.

A2 пишет `kind=llm`, `operation=transcription`, provider/model, статус, задержку,
ID задачи, число опросов, MIME, размер, хеш и доступную длительность. Токены и
цена неизвестны (`usage_known=false`, `cost=null`, `UNKNOWN_USAGE`), а не бесплатны.
Сам клиент Pneuma не пишет текст/аудио/ответы сервера в журнал даже при `LOG_TEXT=true`.
Если текст далее отправлен чат-модели, на её логирование действуют обычные настройки
A1/A2. Хеши, ID пользователей и длительность также могут быть чувствительными.

## Самостоятельная проверка, когда сервер будет доступен

Живая проверка пока не выполнялась по просьбе владельца. Тесты используют
имитацию HTTP и реальные SDK Telegram/VK без обращения к внешним сервисам.

```powershell
# Только тесты, без сети и env-ключей:
uv run --no-sync pytest -q tests/test_pneuma.py tests/test_pneuma_upload.py tests/test_upload_state.py tests/test_transcript_pipeline.py

# Позже: один явно выбранный файл. LLM вызывается только если ASR_POSTPROCESS_ENABLED=true:
uv run --no-sync python examples/transcribe_audio.py voice.mp3 --env .env --live
uv run --no-sync python examples/transcribe_audio.py voice.ogg --env .env --live --output artifacts/transcript.json
```

Каталог вывода создайте заранее. Существующий JSON не перезаписывается. Скрипт
показывает текст в терминале, по `--output` сохраняет его; не публикуйте эти файлы.
Без `--live` скрипт запрещает отправку. `.env.acceptance` можно указать через `--env`.

В `scripts/acceptance.py --mode telegram|vk --live` при включённом Pneuma добавлены
проверки голосового OGG и MP3. Отправьте боту свои тестовые записи; `/status` покажет
непройденные пункты. ASR расходует общий `--max-calls` вместе с LLM/vision.
Этот стенд проверяет непустой транскрипт и учёт, а не точность распознавания.
