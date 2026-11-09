"""Drive the running Java API end to end: register, log in, upload, query, list, and check user isolation.

Needs the Python gRPC server (`main.py serve`) and the Java API running. Real
models, real PostgreSQL, no fakes; queries take minutes on a CPU.

    uv run python scripts/backend_e2e.py --base-url http://localhost:8090
    uv run python scripts/backend_e2e.py --resume STATE.json   # after a restart

It writes two users named e2e_<random>, their conversations, and one small
private document holding a made-up fact, and saves what it needs to resume in
the --state file.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

FACT_DOCUMENT = """# Zephyrine Relay Protocol

## Heartbeats

In the Zephyrine relay protocol every lantern node sends a heartbeat to its
quorum every 17 seconds. A lantern node that misses three consecutive
heartbeats is fenced by the quorum.

## Quorum

A Zephyrine quorum always has exactly five lantern nodes.
"""
FACT_QUESTION = "In the Zephyrine relay protocol, how often does a lantern node send a heartbeat?"
FACT = "17"

QUERY_TIMEOUT = 30 * 60


class Api:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def call(self, method, path, body=None, token=None, headers=None, timeout=60):
        data, sent = None, dict(headers or {})
        if isinstance(body, (dict, list)):
            data = json.dumps(body).encode()
            sent["Content-Type"] = "application/json"
        elif isinstance(body, tuple):
            data, sent["Content-Type"] = body
        if token:
            sent["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(self.base + path, data=data, headers=sent, method=method)
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status, raw, out = response.status, response.read(), dict(response.headers)
        except urllib.error.HTTPError as error:
            status, raw, out = error.code, error.read(), dict(error.headers)
        seconds = time.monotonic() - started
        try:
            parsed = json.loads(raw) if raw else None
        except ValueError:
            parsed = raw.decode(errors="replace")
        return status, parsed, out, seconds


def multipart(files):
    boundary = uuid.uuid4().hex
    parts = []
    for name, data in files:
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class Run:
    def __init__(self, api: Api) -> None:
        self.api = api
        self.results = []

    def check(self, name, ok, detail=""):
        self.results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
        return ok

    def account(self):
        suffix = uuid.uuid4().hex[:8]
        user = {"username": f"e2e_{suffix}", "email": f"e2e.{suffix}@example.com",
                "password": f"e2e-password-{suffix}"}
        status, body, _, _ = self.api.call("POST", "/register", user)
        self.check(f"register {user['username']}", status == 201 and "passwordHash" not in json.dumps(body),
                   f"{status}")
        status, body, _, _ = self.api.call("POST", "/login", {"email": user["email"], "password": user["password"]})
        self.check(f"login {user['username']}", status == 200 and body.get("tokenType") == "Bearer", f"{status}")
        user.update(token=body["accessToken"], user_id=None)
        return user

    def ask(self, user, question, conversation=None, label=""):
        body = {"query": question}
        if conversation:
            body["conversationId"] = conversation
        status, response, headers, seconds = self.api.call("POST", "/query", body, user["token"],
                                                           timeout=QUERY_TIMEOUT)
        print(f"    {label}: HTTP {status} in {seconds:.0f}s", flush=True)
        if status == 200:
            print(f"    answer: {response['answer'][:300]!r}", flush=True)
        return status, response


def listings(run: Run, owner, other, conversation, question, turns) -> None:
    """GET /conversations and /conversations/{id}/exchanges for the owner, and refusals for everyone else."""
    api = run.api
    status, body, _, seconds = api.call("GET", "/conversations", token=owner["token"])
    print(f"    list: HTTP {status} in {seconds:.2f}s", flush=True)
    mine = {c["conversationId"]: c for c in (body or {}).get("conversations", [])} if status == 200 else {}
    run.check("the owner's list includes the conversation", conversation in mine, f"{status}")
    run.check(f"the list counts {turns} exchange(s)", mine.get(conversation, {}).get("exchangeCount") == turns,
              f"{mine.get(conversation, {}).get('exchangeCount')}")
    run.check("the most recently active conversation is listed first",
              status == 200 and body["conversations"] and body["conversations"][0]["conversationId"] == conversation)

    path = f"/conversations/{conversation}/exchanges"
    status, body, _, _ = api.call("GET", path, token=owner["token"])
    exchanges = (body or {}).get("exchanges", []) if status == 200 else []
    run.check(f"the thread has {turns} exchange(s), oldest first", len(exchanges) == turns
              and exchanges[0]["query"] == question
              and [e["askedAt"] for e in exchanges] == sorted(e["askedAt"] for e in exchanges), f"{status}")
    run.check("each exchange carries its answer", exchanges and all(e.get("answer") for e in exchanges))

    status, body, _, _ = api.call("GET", "/conversations", token=other["token"])
    run.check("another user's list does not include it", status == 200
              and conversation not in {c["conversationId"] for c in body["conversations"]}, f"{status}")
    status, body, _, _ = api.call("GET", path, token=other["token"])
    run.check("another user cannot read the thread", status == 404 and body["code"] == "CONVERSATION_NOT_FOUND",
              f"{status}")
    status, _, _, _ = api.call("GET", "/conversations")
    run.check("an unauthenticated list is 401", status == 401, f"{status}")
    status, body, _, _ = api.call("GET", "/conversations?limit=0", token=owner["token"])
    run.check("an out-of-range limit is 400", status == 400 and body["code"] == "VALIDATION_FAILED", f"{status}")
    status, body, _, _ = api.call("GET", "/conversations?cursor=bm90LWEtY3Vyc29y", token=owner["token"])
    run.check("a cursor the API never issued is 400", status == 400 and body["code"] == "VALIDATION_FAILED",
              f"{status}")


def full(run: Run, state_path: str) -> dict:
    api = run.api
    status, body, _, _ = api.call("POST", "/query", {"query": "hi"})
    run.check("unauthenticated query is refused", status == 401 and body["code"] == "UNAUTHORIZED", f"{status}")
    status, _, _, _ = api.call("POST", "/query", {"query": "hi"}, token="not.a.token")
    run.check("invalid token is refused", status == 401, f"{status}")

    alice, bob = run.account(), run.account()
    status, body, _, _ = api.call("POST", "/register", {k: alice[k] for k in ("username", "email", "password")})
    run.check("duplicate registration is a conflict", status == 409, f"{status} {body and body.get('code')}")
    status, body, _, _ = api.call("POST", "/login", {"email": alice["email"], "password": "wrong-password"})
    run.check("wrong password is refused generically", status == 401 and body["code"] == "INVALID_CREDENTIALS")

    status, body, _, seconds = api.call("POST", "/ingest", multipart([
        ("zephyrine.md", FACT_DOCUMENT.encode()), ("tool.exe", b"MZ\x90\x00"), ("fake.pdf", b"not a pdf")]),
        alice["token"], timeout=QUERY_TIMEOUT)
    print(f"    ingest: HTTP {status} in {seconds:.1f}s: {json.dumps(body)[:600]}", flush=True)
    files = {f["filename"]: f for f in (body or {}).get("files", [])}
    run.check("upload is partial: the document in, the bad files out", status == 207
              and files.get("zephyrine.md", {}).get("status") == "INGESTED"
              and files.get("tool.exe", {}).get("errorCode") == "UNSUPPORTED_TYPE"
              and files.get("fake.pdf", {}).get("errorCode") == "MALFORMED", f"{status}")
    document_id = files.get("zephyrine.md", {}).get("documentId")
    status, _, _, _ = api.call("POST", "/ingest", multipart([("x.exe", b"MZ")]), alice["token"])
    run.check("an upload of only unsupported files is 415", status == 415, f"{status}")
    status, _, _, _ = api.call("POST", "/ingest", multipart([("doc.txt", b"text")]))
    run.check("an unauthenticated upload is 401", status == 401, f"{status}")

    status, first = run.ask(alice, FACT_QUESTION, label="alice asks about her document")
    run.check("alice's query completes", status == 200 and first["status"].startswith("COMPLETED"), f"{status}")
    run.check("alice's answer uses her private document", status == 200 and FACT in first["answer"])
    conversation = first.get("conversationId") if status == 200 else None

    status, theirs = run.ask(bob, FACT_QUESTION, label="bob asks the same question")
    run.check("bob's query completes", status == 200, f"{status}")
    run.check("bob's answer does not contain alice's private fact",
              status == 200 and "17 second" not in theirs["answer"] and "every 17" not in theirs["answer"])

    status, body = run.ask(bob, "Continue this conversation", conversation, label="bob uses alice's conversation id")
    run.check("bob cannot use alice's conversation", status == 404 and body["code"] == "CONVERSATION_NOT_FOUND",
              f"{status}")

    if conversation:
        listings(run, alice, bob, conversation, FACT_QUESTION, turns=1)

    state = {"alice": alice, "bob": bob, "conversation": conversation, "document_id": document_id}
    with open(state_path, "w") as handle:
        json.dump(state, handle)
    return state


def resume(run: Run, state_path: str) -> None:
    with open(state_path) as handle:
        state = json.load(handle)
    alice = state["alice"]
    status, body, _, _ = run.api.call("POST", "/login", {"email": alice["email"], "password": alice["password"]})
    run.check("alice can still log in after the restart", status == 200, f"{status}")
    alice["token"] = body["accessToken"]
    status, body = run.ask(alice, "How many lantern nodes are in that quorum?", state["conversation"],
                           label="alice continues her conversation after the restart")
    run.check("alice's conversation survived the restart", status == 200
              and body["conversationId"] == state["conversation"] and not body["newConversation"], f"{status}")

    bob = state["bob"]
    status, body, _, _ = run.api.call("POST", "/login", {"email": bob["email"], "password": bob["password"]})
    run.check("bob can still log in after the restart", status == 200, f"{status}")
    bob["token"] = body["accessToken"]
    listings(run, alice, bob, state["conversation"], FACT_QUESTION, turns=2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--state", default="backend_e2e_state.json")
    parser.add_argument("--resume", action="store_true", help="only the after-restart checks")
    args = parser.parse_args(argv)
    run = Run(Api(args.base_url))
    if args.resume:
        resume(run, args.state)
    else:
        full(run, args.state)
    failed = [name for name, ok, _ in run.results if not ok]
    print(f"\n{len(run.results) - len(failed)} passed, {len(failed)} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
