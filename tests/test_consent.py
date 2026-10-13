import asyncio

import pytest

from botkit.consent.base import CALLBACK_DECLINE, CALLBACK_GRANT, ConsentGate, ConsentPolicy
from botkit.consent.sqlite_store import SQLiteConsentStore

POLICY = ConsentPolicy(version="v1", text="1/ политика\n2/ согласие")


@pytest.fixture
def store(tmp_path):
    return SQLiteConsentStore(str(tmp_path / "consent.sqlite3"))


@pytest.fixture
def gate(store):
    return ConsentGate(store, POLICY)


# --- check() ----------------------------------------------------------------


async def test_new_actor_gets_prompt_with_two_buttons(gate):
    prompt = await gate.check("student-1")

    assert prompt is not None
    assert prompt.text == POLICY.text
    assert [b.text for row in prompt.buttons for b in row] == ["Согласен", "Не согласен"]
    assert [b.data for row in prompt.buttons for b in row] == [CALLBACK_GRANT, CALLBACK_DECLINE]


async def test_granted_actor_passes(gate):
    await gate.handle_callback("student-1", CALLBACK_GRANT)

    assert await gate.check("student-1") is None


async def test_consent_is_per_actor(gate):
    await gate.handle_callback("student-1", CALLBACK_GRANT)

    assert await gate.check("student-2") is not None


# --- handle_callback() ------------------------------------------------------


async def test_grant_returns_granted_without_prompt(gate):
    result = await gate.handle_callback("student-1", CALLBACK_GRANT)

    assert result.granted is True
    assert result.message is None
    assert result.prompt is None


async def test_decline_blocks_and_shows_prompt_again(gate):
    result = await gate.handle_callback("student-1", CALLBACK_DECLINE)

    assert result.granted is False
    assert result.message == POLICY.declined_text
    assert result.prompt is not None
    assert await gate.check("student-1") is not None


async def test_grant_after_decline_lets_actor_in(gate):
    await gate.handle_callback("student-1", CALLBACK_DECLINE)
    await gate.handle_callback("student-1", CALLBACK_GRANT)

    assert await gate.check("student-1") is None


async def test_handle_callback_rejects_foreign_data(gate):
    with pytest.raises(ValueError):
        await gate.handle_callback("student-1", "uv:UV02")


def test_is_consent_callback():
    assert ConsentGate.is_consent_callback(CALLBACK_GRANT)
    assert ConsentGate.is_consent_callback(CALLBACK_DECLINE)
    assert not ConsentGate.is_consent_callback("uv:UV02")


async def test_double_click_is_harmless(gate):
    await asyncio.gather(*[gate.handle_callback("student-1", CALLBACK_GRANT) for _ in range(5)])

    assert await gate.check("student-1") is None


# --- версия текста ----------------------------------------------------------


async def test_new_policy_version_asks_again(store):
    await ConsentGate(store, POLICY).handle_callback("student-1", CALLBACK_GRANT)
    newer = ConsentPolicy(version="v2", text="новый текст")

    assert await ConsentGate(store, newer).check("student-1") is not None


# --- отзыв оператором (без отдельного сценария) -------------------------------


async def test_revoked_decision_blocks_again(store, gate):
    await gate.handle_callback("student-1", CALLBACK_GRANT)

    await store.record(
        actor_id="student-1", policy_version="v1", text_sha256=POLICY.text_sha256, decision="revoked"
    )

    assert await gate.check("student-1") is not None


async def test_record_rejects_unknown_decision(store):
    with pytest.raises(ValueError):
        await store.record(actor_id="s", policy_version="v1", text_sha256="x", decision="maybe")


# --- журнал -----------------------------------------------------------------


async def test_journal_is_append_only_and_keeps_text_hash(store, gate):
    await gate.handle_callback("student-1", CALLBACK_DECLINE)
    await gate.handle_callback("student-1", CALLBACK_GRANT)

    events = await store.list_events("student-1")

    assert [e["decision"] for e in events] == ["declined", "granted"]
    assert {e["text_sha256"] for e in events} == {POLICY.text_sha256}
    assert {e["policy_version"] for e in events} == {"v1"}
    assert all(e["created_at"] for e in events)


async def test_data_survives_reopening_the_store(tmp_path):
    path = str(tmp_path / "consent.sqlite3")
    await ConsentGate(SQLiteConsentStore(path), POLICY).handle_callback("student-1", CALLBACK_GRANT)

    assert await ConsentGate(SQLiteConsentStore(path), POLICY).check("student-1") is None
