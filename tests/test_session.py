# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for gepa.core.session — Session protocol and implementations."""

from __future__ import annotations

from gepa.core.session import MessageListSession, NullSession, Session, make_session_lm


class TestMessageListSession:
    def _make_echo_session(self, system_prompt: str = "You are helpful.") -> MessageListSession:
        """Create a session with a simple echo API call."""

        def echo_api(messages: list[dict], **kwargs) -> str:  # noqa: ARG001
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            return f"echo: {last_user}"

        return MessageListSession(system_prompt=system_prompt, api_call=echo_api)

    def test_protocol_compliance(self) -> None:
        session = self._make_echo_session()
        assert isinstance(session, Session)

    def test_send_appends_messages(self) -> None:
        session = self._make_echo_session()
        response = session.send("hello")
        assert response == "echo: hello"
        assert len(session.history) == 2
        assert session.history[0] == {"role": "user", "content": "hello"}
        assert session.history[1] == {"role": "assistant", "content": "echo: hello"}

    def test_send_multiple(self) -> None:
        session = self._make_echo_session()
        session.send("first")
        session.send("second")
        assert len(session.history) == 4

    def test_fork_copies_history(self) -> None:
        session = self._make_echo_session()
        session.send("before fork")
        forked = session.fork("child")

        assert forked.session_id != session.session_id
        assert "child" in forked.session_id
        assert forked.history == session.history

    def test_fork_diverges(self) -> None:
        session = self._make_echo_session()
        session.send("shared")
        forked = session.fork()

        session.send("parent only")
        forked.send("child only")

        assert len(session.history) == 4
        assert len(forked.history) == 4
        assert session.history[-1]["content"] == "echo: parent only"
        assert forked.history[-1]["content"] == "echo: child only"

    def test_reset_clears_history(self) -> None:
        session = self._make_echo_session()
        session.send("hello")
        assert len(session.history) > 0
        session.reset()
        assert len(session.history) == 0

    def test_reset_preserves_system_prompt(self) -> None:
        session = self._make_echo_session(system_prompt="custom prompt")
        session.send("hello")
        session.reset()
        response = session.send("after reset")
        assert response == "echo: after reset"
        assert len(session.history) == 2

    def test_history_is_copy(self) -> None:
        session = self._make_echo_session()
        session.send("hello")
        history = session.history
        history.clear()
        assert len(session.history) == 2

    def test_session_id_auto_generated(self) -> None:
        session = self._make_echo_session()
        assert len(session.session_id) > 0

    def test_session_id_explicit(self) -> None:
        def noop(messages, **kwargs):  # noqa: ARG001
            return ""

        session = MessageListSession(system_prompt="", api_call=noop, session_id="my-id")
        assert session.session_id == "my-id"

    def test_api_call_receives_system_prompt(self) -> None:
        received_messages = []

        def capture_api(messages: list[dict], **kwargs) -> str:  # noqa: ARG001
            received_messages.extend(messages)
            return "ok"

        session = MessageListSession(system_prompt="Be concise.", api_call=capture_api)
        session.send("test")
        assert received_messages[0] == {"role": "system", "content": "Be concise."}


class TestNullSession:
    def test_protocol_compliance(self) -> None:
        session = NullSession()
        assert isinstance(session, Session)

    def test_send_returns_empty(self) -> None:
        session = NullSession()
        assert session.send("anything") == ""

    def test_history_empty(self) -> None:
        session = NullSession()
        session.send("something")
        assert session.history == []

    def test_fork_returns_new_null(self) -> None:
        session = NullSession()
        forked = session.fork("label")
        assert isinstance(forked, NullSession)
        assert forked.session_id != session.session_id

    def test_reset_noop(self) -> None:
        session = NullSession()
        session.reset()  # should not raise


class TestMakeSessionLm:
    def _make_echo_session(self) -> MessageListSession:
        def echo_api(messages: list[dict], **kwargs) -> str:  # noqa: ARG001
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            return f"echo: {last_user}"

        return MessageListSession(system_prompt="test", api_call=echo_api)

    def test_string_prompt(self) -> None:
        session = self._make_echo_session()
        lm = make_session_lm(session)
        result = lm("hello")
        assert result == "echo: hello"

    def test_message_list_prompt(self) -> None:
        session = self._make_echo_session()
        lm = make_session_lm(session)
        result = lm([{"role": "user", "content": "from list"}])
        assert result == "echo: from list"

    def test_session_accumulates_history(self) -> None:
        session = self._make_echo_session()
        lm = make_session_lm(session)
        lm("first")
        lm("second")
        assert len(session.history) == 4  # 2 user + 2 assistant

    def test_callable_satisfies_lm_protocol(self) -> None:
        session = self._make_echo_session()
        lm = make_session_lm(session)
        assert callable(lm)
