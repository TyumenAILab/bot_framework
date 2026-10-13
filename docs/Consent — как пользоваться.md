# Consent: как пользоваться

Практическое руководство по `botkit.consent` — коду в `src/botkit/consent/`. Модуль не соответствует SC-контракту и вне нумерации A/B/C.

Задача: бот хранит id пользователя мессенджера (например, VK), поэтому при **первом контакте** показывает один экран согласия на обработку персональных данных с кнопками «Согласен» / «Не согласен». Без согласия работа с ботом не продолжается; факт согласия фиксируется.

Модуль намеренно простой: один экран, никакого меню настроек и команды отзыва. Отзыв делает оператор (преподаватель) записью в тот же журнал — см. ниже.

---

## Общая схема

```
ConsentPolicy(version, text, ...)        — что показываем (данные бота)
SQLiteConsentStore(db_path)              — журнал решений, только дописывается
ConsentGate(store, policy)               — логика
   │
   ├── check(actor_id)            → None (можно) | ConsentPrompt (показать экран)
   ├── is_consent_callback(data)  → это нажатие нашей кнопки?
   └── handle_callback(actor_id, data) → ConsentResult(granted, message, prompt)
```

Модуль **ничего не отправляет сам**: он возвращает текст и кнопки, а отправляет бот через свой адаптер (как и остальные модули фреймворка).

---

## Что во фреймворке, что в боте

| Во фреймворке (`botkit.consent`) | В боте |
|---|---|
| Журнал решений, статус по последней записи | Текст экрана, ссылки, номер версии (данные оператора) |
| Проверка «есть ли согласие на **текущую** версию текста» | Вызов `check()` в начале **каждого** обработчика |
| Две кнопки и обработка их нажатий | Отправка экрана через адаптер |
| Ответ на отказ: «дальше нельзя» + экран заново | Что показать после согласия (меню, приветствие) |
| | Какой `actor_id` передавать (напр. обезличенный) |
| | Отзыв согласия оператором и что делать с данными после отзыва |

---

## Шаг 1 — политика и стор

```python
from botkit.consent.base import ConsentGate, ConsentPolicy
from botkit.consent.sqlite_store import SQLiteConsentStore

POLICY = ConsentPolicy(
    version="2026-10",   # поменяли текст или ссылки — поменяйте и версию
    text=(
        "Для работы с ботом нужно согласие.\n\n"
        "1/ Подтверждаю ознакомление с политикой оператора в отношении обработки "
        "персональных данных в ТюмГУ https://www.utmn.ru/o-tyumgu/soglasheniya/politika-konfidentsialnosti.php\n\n"
        "2/ Подтверждаю согласие на обработку персональных данных "
        "https://www.utmn.ru/o-tyumgu/soglasheniya/soglasie/\n\n"
        "Нажимая «Согласен», вы подтверждаете оба пункта."
    ),
)

consent = ConsentGate(SQLiteConsentStore("data/consent.sqlite3"), POLICY)
```

Файл и таблица `consent_events` создаются конструктором. Можно указать тот же файл БД, что у других сторов бота.

Остальные поля `ConsentPolicy` имеют значения по умолчанию: `grant_label="Согласен"`, `decline_label="Не согласен"`, `declined_text` — сообщение «дальше пройти нельзя». Тексты кнопок для VK — не длиннее 40 символов.

## Шаг 2 — подключить к обработчикам

Бот вешает обработчики вручную (общего слоя, через который проходят все сообщения, во фреймворке нет), поэтому проверка вызывается в каждом из них — **и в сообщениях, и в callback'ах, и во вложениях**:

```python
async def show_consent(chat_id: str, prompt) -> None:
    await adapter.send(chat_id, prompt.text, buttons=prompt.buttons)


@adapter.on_command("start")
async def on_start(message: IncomingMessage) -> None:
    actor_id = actor(message.platform, message.user_id)
    prompt = await consent.check(actor_id)
    if prompt is not None:
        await show_consent(message.chat_id, prompt)
        return
    ...  # обычный старт бота


@adapter.on_message
async def on_message(message: IncomingMessage) -> None:
    actor_id = actor(message.platform, message.user_id)
    prompt = await consent.check(actor_id)
    if prompt is not None:          # любое сообщение до согласия → экран согласия
        await show_consent(message.chat_id, prompt)
        return
    ...


@adapter.on_callback
async def on_callback(query: CallbackQuery) -> None:
    actor_id = actor(query.platform, query.user_id)

    if consent.is_consent_callback(query.data):      # наши кнопки — мимо проверки
        result = await consent.handle_callback(actor_id, query.data)
        if result.granted:
            ...                                       # меню / приветствие бота
        else:
            await adapter.send(query.chat_id, result.message)
            await show_consent(query.chat_id, result.prompt)
        return

    prompt = await consent.check(actor_id)            # кнопки из старых сообщений тоже за гейтом
    if prompt is not None:
        await show_consent(query.chat_id, prompt)
        return
    ...
```

Проверка в callback'е обязательна: иначе «не согласен» обходится нажатием кнопки из более раннего сообщения.

Проверено прогоном через `MemoryAdapter`: `adapter.send(chat_id, prompt.text, buttons=prompt.buttons)` принимает готовые кнопки.

## Как работает

- **Нет решения / отказ / отзыв** — одинаково «не пускать»: `check()` вернёт экран.
- **«Не согласен»** → `ConsentResult(granted=False, message=..., prompt=...)`: сообщение «дальше пройти нельзя» и экран согласия заново. Отказ записывается в журнал.
- **«Согласен» после отказа** работает: статус определяется последней записью.
- **Смена `version`** → все получают экран заново; согласие на старую версию не считается.
- **Двойной клик** безвреден: просто две записи `granted`.

## Журнал и отзыв оператором

Таблица `consent_events` только дописывается: `actor_id`, `policy_version`, `text_sha256` (отпечаток показанного текста — если текст поправили, не меняя версию, это видно), `decision` (`granted` / `declined` / `revoked`), `created_at`.

```python
await store.list_events(actor_id)       # вся история решений актора, по порядку
await store.current(actor_id, "2026-10")   # "granted" | "declined" | "revoked" | None
```

Отзыв отдельного сценария не имеет: преподаватель (через ваш кабинет/команду) записывает

```python
await store.record(
    actor_id=actor_id, policy_version=POLICY.version,
    text_sha256=POLICY.text_sha256, decision="revoked",
)
```

После этого `check()` снова вернёт экран согласия. Удаление данных студента после отзыва модуль **не делает** — это решение и код бота.

## Ограничения

- Согласие общее, один раз на бота и версию текста; по пунктам (политика / согласие) не разделяется. Если оператору нужны раздельные подтверждения, это изменение модуля.
- Если вы обезличиваете id (как `psychologic_trainer`), передавайте в `check`/`handle_callback` тот же `actor_id`, что и в остальной код бота; сам модуль id не преобразует.
- До согласия не пишите id пользователя в логи и БД бота, если согласие нужно именно для его хранения: в журнале согласия единственная запись о нём — само решение.
