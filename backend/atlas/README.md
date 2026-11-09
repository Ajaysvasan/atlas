# Atlas API (Java / Spring Boot)

The public REST API of Atlas and its security boundary. It owns accounts, password hashing,
JWTs, request validation and uploads. It passes queries, ingestion and conversation listings
to the Python pipeline ([`../app`](../app/README.md)) over gRPC, and turns the pipeline's
errors into HTTP responses. The Java service itself has no RAG logic.

```text
web frontend ──REST──▶ this API (:8080) ──gRPC + service token──▶ Python pipeline (127.0.0.1:50051)
                            │
                            └── PostgreSQL: user_table (Flyway)
```

[← Project README](../../README.md) · [Backend reference: settings, REST API, security](../BACKEND.md) ·
[Web frontend](../../frontend/web_frontend/README.md) · [Python pipeline](../app/README.md)

## Stack

Java 21, Spring Boot 4 (Web MVC, Security with an OAuth2 resource server for JWTs, JDBC,
Validation, Actuator), Flyway, PostgreSQL, grpc-java with protobuf (bindings generated from
[`../proto`](../proto/atlas/v1/atlas_service.proto) at build time), BouncyCastle Argon2id,
and Bucket4j with Caffeine for rate limits.
Tests use JUnit 5, Testcontainers PostgreSQL and an in-process fake gRPC pipeline.

## Endpoints

| Method and path | Auth | Purpose |
|---|---|---|
| `POST /register` | public | Create an account |
| `POST /login` | public | Exchange email and password for a Bearer JWT |
| `POST /query` | Bearer | Ask a question in a new or existing conversation |
| `POST /ingest` | Bearer | Upload documents (multipart `files`), private to the uploader |
| `GET /conversations` | Bearer | The caller's conversations, most recently active first, paged |
| `GET /conversations/{id}/exchanges` | Bearer | One conversation's questions and answers, oldest first, paged |
| `GET /actuator/health` | public | Whether the Java service is up |

[`../BACKEND.md`](../BACKEND.md#rest-api) has the request and response bodies, every status
code and the error format.

## Running

Prerequisites: JDK 21, the PostgreSQL database set up for the pipeline (see the
[project README](../../README.md#1-database)), and the Python pipeline running
(`main.py serve`).

```bash
source ~/.config/atlas/secrets.env     # ATLAS_JWT_SECRET and ATLAS_GRPC_SERVICE_TOKEN, shared with Python
./mvnw spring-boot:run                 # or: ./mvnw package && java -jar target/atlas-*.jar
```

- **Required:** both `ATLAS_JWT_SECRET` and `ATLAS_GRPC_SERVICE_TOKEN`. The service refuses
  to start without them.
- **Same token:** the token must be the same value the Python server was started with, or
  every pipeline call fails with `BACKEND_UNAVAILABLE`.
- **Database settings:** read from the `DB_*` keys in `../app/.env`. Override them with
  `ATLAS_DB_URL`, `ATLAS_DB_USER` and `ATLAS_DB_PASSWORD`, or point `ATLAS_ENV_FILE`
  elsewhere.
- **Port:** the default is `8080`. Set `ATLAS_HTTP_PORT=8081` if something else holds 8080.
- **CORS:** allows `http://localhost:5173` and `http://localhost:3000` by default. Change it
  with `ATLAS_ALLOWED_ORIGINS`.
- **Other settings** (deadlines, TLS or mTLS to the pipeline, the Argon2 work factor, upload
  and rate limits) are listed in [`../BACKEND.md`](../BACKEND.md#configuration).

The default values are in
[`src/main/resources/application.properties`](src/main/resources/application.properties).

On first start Flyway creates `user_table` (see
[`V1__create_user_table.sql`](src/main/resources/db/migration/V1__create_user_table.sql)).
It records its history in `atlas_flyway_history` and never runs `clean`.

## Code layout (`src/main/java/com/fyp/atlas`)

| Package | Contents |
|---|---|
| `controller/` | `AuthController` (register, login), `PipelineController` (query, ingest), `ConversationController` (listings) |
| `service/` | Business logic: `AuthService`, `QueryService`, `IngestService`, `ConversationService`, `UploadValidator` (type, size and content checks), and `QResponseHandler` (checks the pipeline's answer and builds the REST body) |
| `grpc/` | `AtlasGrpcClient` (deadlines, service token), `GrpcChannelConfig` (plaintext, TLS or mTLS), `GrpcErrorTranslator` (gRPC status and `atlas-error-code` → `ApiException`) |
| `security/` | `SecurityConfig`, `JwtTokenService` (HS256), `PasswordConfig` (Argon2id), `RateLimiter`, and the JSON 401/403 handlers |
| `exception/` | `ErrorCode`, `ApiException`, and `GlobalExceptionHandler`, which gives every error one JSON shape |
| `web/` | `RequestCorrelationFilter`: accepts or creates `X-Request-Id` and passes it on to Python |
| `dto/`, `entity/`, `repository/` | Request and response records, `UserAccount`, and `UserRepository` |
| `config/` | `AtlasProperties` (typed, validated `atlas.*` settings) and the clock |

## Tests

```bash
./mvnw test
```

- **Docker required:** Testcontainers starts a throwaway PostgreSQL, so the tests never touch
  the database in `../app/.env`.
- **Fake pipeline:** an in-process fake gRPC pipeline (`support/FakePipeline`) stands in for
  Python.
- **Coverage:** authentication, the query, ingest and conversation endpoints, the gRPC
  boundary (deadlines, error translation, the service token), JWTs, password hashing, rate
  limits, upload rules and the migration.

To test against the real pipeline, start both services and run the end-to-end driver:

```bash
cd ../app && uv run --no-sync python scripts/backend_e2e.py --base-url http://localhost:8080
```

It creates throwaway `e2e_*` users and one small private document.
