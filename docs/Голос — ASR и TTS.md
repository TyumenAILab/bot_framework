# Голос — ASR и TTS

Два независимых модуля фреймворка: `botkit.asr` (голос → текст) и `botkit.tts`
(текст → голос) плюс отправка голосовых сообщений в VK. Логика перенесена из бота
SocratiQ, но вынесена за интерфейсы: **модель или сервис меняется строкой в `.env`
или своим классом, код бота не трогается.**

```
VK audio_message ─► Attachment ─► transcribe_attachment(asr, …) ─► текст ─► ваша логика / LLM
                                         │
                       ASR_PROVIDER: pneuma | moonshine | module:factory

ответ (текст) ─► speak(adapter, chat_id, текст, tts) ─► VKAdapter.send_attachment(kind="voice")
                          │
        TTS_PROVIDER: yandex | google | kokoro | module:factory
```

## ASR — распознавание

**Интерфейс** — уже существовавший в A1 `AudioTranscriptionProvider`:

```python
async def atranscribe(data: bytes, *, mime_type: str, filename: str,
                      timeout: float | None = None, language: str | None = None) -> LLMResponse
```

`result.response` — итоговый текст, `result.raw` — текст движка как есть,
`result.meta["audio_duration_seconds"]` — длительность, если движок её знает.
Поскольку интерфейс общий, любой провайдер без доработок работает и в A5
(`ContentExtractionService`): голосовые вложения разбираются тем же движком.

| `ASR_PROVIDER` | Что это | Настройки |
|---|---|---|
| `pneuma` | Очередь Pneuma (legacy / chunked) | [Pneuma ASR](Pneuma%20ASR.md) |
| `moonshine` | Свой сервер `POST /v1/audio/transcriptions` | `MOONSHINE_API_URL`, `MOONSHINE_API_KEY`, `MOONSHINE_LANGUAGE`, `MOONSHINE_TIMEOUT`, `MOONSHINE_MAX_CONCURRENT` |
| `module:factory` | Ваш класс | см. ниже |
| `disabled` (по умолчанию) | ASR выключен, `load_asr()` вернёт `None` | — |

```python
from botkit.asr import load_asr, transcribe_attachment

asr = load_asr(tracker=tracker)          # по ASR_PROVIDER; None, если выключено
result = await transcribe_attachment(asr, message.attachments[0], timeout=120)
text = result.response
...
await asr.aclose()                       # при остановке бота
```

`transcribe_attachment` скачивает вложение (с лимитом размера адаптера), подбирает
имя файла по MIME и убирает префиксы `[00:00:00.000 - …] Speaker 1:`, которые
добавляет Pneuma. Для Moonshine можно передать `on_queued=` напрямую в `atranscribe`:
колбэк вызовется один раз, если все слоты сервера заняты («перед вами N записей»).

`ASR_POSTPROCESS_ENABLED=true` (восстановление пунктуации локальной LLM) работает
с любым встроенным провайдером — см. [Pneuma ASR](Pneuma%20ASR.md).

## TTS — озвучка

**Интерфейс:**

```python
class SpeechSynthesizer(Protocol):
    async def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio
    async def aclose(self) -> None

SynthesizedAudio(data: bytes, mime_type="audio/ogg", seconds: float | None = None)
```

`voice` необязателен и переопределяет голос из настроек. `seconds` — длительность
звучания, если провайдер её сообщает (полезно, чтобы не считать прослушивание «временем
на размышление»).

| `TTS_PROVIDER` | Что это | Настройки |
|---|---|---|
| `yandex` | Yandex SpeechKit REST v1 | `YANDEX_TTS_API_KEY` **или** `YANDEX_TTS_IAM_TOKEN`, `YANDEX_TTS_VOICE`, `…_LANGUAGE`, `…_SPEED`, `…_FOLDER_ID` (только для IAM пользователя) |
| `google` | Google Cloud TTS | ADC (`GOOGLE_APPLICATION_CREDENTIALS`, extra `botkit[google]`) или `GOOGLE_TTS_ACCESS_TOKEN`; `GOOGLE_TTS_VOICE/LANGUAGE/SPEED/QUOTA_PROJECT` |
| `kokoro` | Свой сервер Kokoro (английский) | `KOKORO_TTS_URL`, `…_API_KEY`, `…_VOICE`, `…_SPEED`, `…_OUTPUT`, `…_OPUS_RATE`, `…_MAX_CONCURRENT`, `…_ATTEMPTS`, `…_ATTEMPT_TIMEOUT`, `…_IDLE_TIMEOUT` |
| `module:factory` | Ваш класс | см. ниже |
| `disabled` (по умолчанию) | TTS выключен, `load_tts()` вернёт `None` | — |

Общие: `TTS_FORMAT` (`oggopus` — голосовое сообщение, `mp3` — аудиофайл),
`TTS_TIMEOUT` (общий бюджет одного синтеза, 6 с), `TTS_CACHE_TTL` (кеш в RAM, 300 с, `0` — выкл.).

```python
from botkit.tts import load_tts, speak

tts = load_tts(tracker=tracker)          # по TTS_PROVIDER; None, если выключено
sent = await speak(adapter, chat_id, "Ваш вопрос…", tts, caption="Вопрос (аудио)")
if sent is None:                         # озвучить/доставить не удалось (причина в логе)
    await adapter.send(chat_id, "Ваш вопрос…")      # запасной вариант — текст
else:
    sent.seconds                         # сколько звучит, если известно
```

`load_tts()` оборачивает провайдера в `CachedSynthesizer`:

- общий таймаут (включая авторизацию и повторы провайдера);
- одинаковые одновременные запросы объединяются в один платный вызов;
- успешный звук хранится **только в RAM** (TTL, не более 64 записей / 20 МиБ); ошибки не кешируются;
- событие A2: `kind=speech`, `operation=tts`, статус, задержка, число символов и байт,
  без текста и звука. **Стоимость не оценивается** (`cost_status=UNKNOWN_USAGE`) —
  прайс-лист токенов не подходит для тарифов речи; `null` не означает «бесплатно».

Для Kokoro и любых провайдеров, возвращающих WAV, нужен `pip install 'botkit[audio]'`:
звук конвертируется в OGG/Opus 16 кГц (VK принимал 16 кГц и отвергал 24/48 кГц на
замере; значение настраивается `KOKORO_TTS_OPUS_RATE`).

## Голосовые сообщения в VK

- **Входящие.** `VKAdapter` уже превращает `audio_message` в `Attachment(kind="audio",
  mime_type="audio/ogg")`. Бот не получает поток с микрофона — только готовую запись, поэтому
  определение паузы «пользователь замолчал» на стороне бота невозможно.
- **Исходящие.** `adapter.send_attachment(chat_id, data, filename="q.ogg", kind="voice")`
  отправляет нативное голосовое (`audio_message`): `docs.getMessagesUploadServer(type=audio_message)`
  → загрузка → `docs.save`. Нужен **OGG/Opus**. Сервер загрузки VK иногда отвечает HTML-страницей
  «temporarily unavailable»; каждая из 3 попыток берёт новый сервер, после неудачи бросается
  `VKUploadError` (без содержимого ответа). `speak()` / `send_audio()` сами выбирают
  `voice` для OGG и `audio` (документ) для MP3/WAV. Telegram с `kind="voice"` работает так же.

Готовый пример — [examples/voice_bot.py](../examples/voice_bot.py): голосовое → ASR → LLM →
текст + голосовой ответ.

## Свой провайдер

Достаточно класса с нужным методом — наследовать ничего не нужно.

```python
# my_pkg/speech.py
from botkit.llm import LLMResponse
from botkit.tts import SynthesizedAudio

class MyASR:
    async def atranscribe(self, data, *, mime_type, filename, timeout=None, language=None):
        text = await my_engine(data)
        return LLMResponse(text, text, "raw_text_fallback", provider="my-asr")
    async def aclose(self): ...

class MyTTS:
    async def synthesize(self, text, *, voice=None):
        return SynthesizedAudio(await my_engine(text), "audio/ogg")
    async def aclose(self): ...

def build_asr(tracker=None): return MyASR()
def build_tts(): return MyTTS()
```

```dotenv
ASR_PROVIDER=my_pkg.speech:build_asr
TTS_PROVIDER=my_pkg.speech:build_tts
```

или в коде: `register_asr("my", build_asr)`, `register_tts("my", lambda timeout: MyTTS())`
и затем `ASR_PROVIDER=my`. Путь `module:factory` задаёт только оператор — никогда не
принимайте его от пользователя.

## Приватность и ограничения

- Аудио и текст не пишутся в журнал и в БД фреймворка, ключи не попадают в ошибки.
  Это не означает отсутствия копий у ASR/TTS-сервиса и у мессенджера — политику хранения
  у этих сторон согласуйте отдельно. Yandex/Google получают текст озвучиваемых реплик.
- Автоматических повторов платных облачных запросов нет (Kokoro повторяет попытки к
  собственному серверу), редиректы запрещены, TLS проверяется.
- Moonshine/Kokoro рассчитаны на частную сеть: переменные прокси из окружения игнорируются.
- Лимиты: вложение до `max_attachment_bytes` адаптера (20 МиБ по умолчанию), звук TTS до 10 МиБ,
  текст — по лимиту провайдера (Yandex 5000 символов, Google 5000 байт, Kokoro 5000 символов).

## Проверка

```bash
uv run --no-sync pytest -q tests/test_asr.py tests/test_tts.py tests/test_vk_voice.py
```

Тесты используют имитацию HTTP и VK API. **Реальные серверы ASR/TTS и живая отправка в VK
не проверялись** — перед пилотом отправьте тестовое голосовое и проверьте ответ в VK.
