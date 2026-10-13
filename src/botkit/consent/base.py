# Consent — одно общее согласие на обработку персональных данных в начале бота.
# Не соответствует SC-контракту; вне нумерации A/B/C. Описание и пример
# подключения: docs/Consent — как пользоваться.md.
#
# Намеренно простой модуль: экран согласия показывается при первом контакте,
# решение пишется в append-only журнал, дальше бот работает как обычно. Нет
# отдельной команды отзыва — отзыв делает оператор (преподаватель) записью
# decision="revoked" в тот же журнал через ConsentStore.record().

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Protocol

from botkit.transport import Button

Decision = Literal["granted", "declined", "revoked"]

CALLBACK_GRANT = "consent:yes"
CALLBACK_DECLINE = "consent:no"


@dataclass(frozen=True)
class ConsentPolicy:
    """Что показываем студенту. Целиком данные бота/оператора, не фреймворка.

    version — явная строка ("2026-10"): при смене текста или ссылок её нужно
    поменять, и все студенты получат экран согласия заново. Согласие на старую
    версию текущей не считается."""

    version: str
    text: str  # весь текст экрана: пункты, ссылки, просьба нажать кнопку
    declined_text: str = (
        "Без согласия на обработку персональных данных дальше пройти нельзя. "
        "Чтобы продолжить, нажмите «Согласен»."
    )
    grant_label: str = "Согласен"
    decline_label: str = "Не согласен"

    @property
    def text_sha256(self) -> str:
        """Отпечаток показанного текста — фиксируется в журнале рядом с версией:
        если текст поправили, не меняя version, расхождение видно по журналу."""
        return hashlib.sha256(self.text.encode()).hexdigest()


@dataclass(frozen=True)
class ConsentPrompt:
    """Готово к отправке: adapter.send(chat_id, prompt.text, buttons=prompt.buttons)."""

    text: str
    buttons: list[list[Button]]


@dataclass(frozen=True)
class ConsentResult:
    granted: bool  # True — можно продолжать работу с ботом
    message: str | None = None  # что сказать перед экраном (при отказе)
    prompt: ConsentPrompt | None = None  # экран согласия, который нужно показать снова


class ConsentStore(Protocol):
    async def record(
        self, *, actor_id: str, policy_version: str, text_sha256: str, decision: Decision
    ) -> None: ...

    async def current(self, actor_id: str, policy_version: str) -> Decision | None:
        """Последнее решение по этой версии текста; None — решения ещё не было."""
        ...

    async def list_events(self, actor_id: str) -> list[dict]: ...


class ConsentGate:
    def __init__(self, store: ConsentStore, policy: ConsentPolicy) -> None:
        self._store = store
        self._policy = policy

    def prompt(self) -> ConsentPrompt:
        return ConsentPrompt(
            text=self._policy.text,
            buttons=[
                [
                    Button(self._policy.grant_label, data=CALLBACK_GRANT),
                    Button(self._policy.decline_label, data=CALLBACK_DECLINE),
                ]
            ],
        )

    async def check(self, actor_id: str) -> ConsentPrompt | None:
        """None — согласие на текущую версию есть, пропускайте сообщение дальше.
        Иначе — экран согласия: покажите его вместо обычной обработки.
        Отсутствие решения, отказ и отзыв одинаково означают «не пускать»."""
        if await self._store.current(actor_id, self._policy.version) == "granted":
            return None
        return self.prompt()

    @staticmethod
    def is_consent_callback(data: str) -> bool:
        return data in (CALLBACK_GRANT, CALLBACK_DECLINE)

    async def handle_callback(self, actor_id: str, data: str) -> ConsentResult:
        """Фиксирует решение. Только для data, прошедших is_consent_callback()."""
        if data == CALLBACK_GRANT:
            decision: Decision = "granted"
        elif data == CALLBACK_DECLINE:
            decision = "declined"
        else:
            raise ValueError(f"not a consent callback: {data!r}")
        await self._store.record(
            actor_id=actor_id,
            policy_version=self._policy.version,
            text_sha256=self._policy.text_sha256,
            decision=decision,
        )
        if decision == "granted":
            return ConsentResult(granted=True)
        return ConsentResult(granted=False, message=self._policy.declined_text, prompt=self.prompt())
