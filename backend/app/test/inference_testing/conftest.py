from config import Config

SYSTEM = (
    "You are a database tutor. Answer in at most two sentences. "
    + "Prefer precise terminology over analogies. " * 12
)


def history(*pairs):
    messages = [{"role": "system", "content": SYSTEM}]
    for user, assistant in pairs:
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return messages


__all__ = ["Config", "SYSTEM", "history"]
