from typing import Mapping, Protocol, Sequence

class ChatMessage(Protocol):
    content: str

class ChatChoice(Protocol):
    message: ChatMessage

class ChatCompletion(Protocol):
    choices: Sequence[ChatChoice]

MessageDict = Mapping[str, str]

def completion(
    *,
    model: str,
    messages: Sequence[MessageDict],
    **kwargs: object,
) -> ChatCompletion: ...
def batch_completion(
    *,
    model: str,
    messages: Sequence[Sequence[MessageDict]],
    max_workers: int | None = ...,
    **kwargs: object,
) -> Sequence[ChatCompletion]: ...
