package com.fyp.atlas.controller;

import static com.fyp.atlas.support.ApiTestSupport.freshAddress;
import static com.fyp.atlas.support.ApiTestSupport.from;
import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.util.Map;
import java.util.UUID;

import com.fyp.atlas.support.ApiTestSupport;
import com.fyp.atlas.support.PostgresContainer;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.context.annotation.Import;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.http.MediaType;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.security.oauth2.jwt.JwtDecoder;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

@SpringBootTest
@AutoConfigureMockMvc
@Import(PostgresContainer.class)
class AuthIntegrationTest {

	@Autowired
	MockMvc mvc;

	@Autowired
	JsonMapper json;

	@Autowired
	JdbcClient jdbc;

	@Autowired
	JwtDecoder decoder;

	ApiTestSupport api;

	@BeforeEach
	void setUp() {
		api = new ApiTestSupport(mvc, json);
	}

	static Map<String, String> registration(String username, String email, String password) {
		return Map.of("username", username, "email", email, "password", password);
	}

	static String unique() {
		return UUID.randomUUID().toString().substring(0, 8);
	}

	@Test
	void registrationStoresAnArgon2idHashAndReturnsNoSecret() throws Exception {
		String suffix = unique();
		MvcResult result = mvc
			.perform(api.jsonPost("/register",
					registration("Dev_" + suffix, "  Dev." + suffix + "@Example.COM ", "example-password")))
			.andExpect(status().isCreated())
			.andExpect(jsonPath("$.username").value("Dev_" + suffix))
			.andExpect(jsonPath("$.email").value("dev." + suffix + "@example.com"))
			.andExpect(jsonPath("$.userId").isNotEmpty())
			.andExpect(jsonPath("$.createdAt").isNotEmpty())
			.andReturn();
		String body = result.getResponse().getContentAsString();
		assertThat(body).doesNotContain("example-password").doesNotContainIgnoringCase("hash")
			.doesNotContain("argon");

		UUID userId = UUID.fromString(api.body(result).get("userId").asString());
		Map<String, Object> row = jdbc.sql("select * from user_table where user_id = ?").param(userId).query().singleRow();
		assertThat(row.get("password_hash").toString()).startsWith("$argon2id$v=19$m=65536,t=3,p=1$")
			.doesNotContain("example-password");
		assertThat(row.get("email")).isEqualTo("dev." + suffix + "@example.com");
		assertThat(row.get("is_active")).isEqualTo(true);
		assertThat(row.get("created_at")).isNotNull();
	}

	@Test
	void duplicateUsernameIgnoringCaseIsAConflict() throws Exception {
		String suffix = unique();
		mvc.perform(api.jsonPost("/register", registration("Taken_" + suffix, "a" + suffix + "@example.com", "password-123")))
			.andExpect(status().isCreated());
		mvc.perform(api.jsonPost("/register", registration("taken_" + suffix, "b" + suffix + "@example.com", "password-123")))
			.andExpect(status().isConflict())
			.andExpect(jsonPath("$.code").value("USERNAME_TAKEN"));
	}

	@Test
	void duplicateEmailIgnoringCaseIsAConflict() throws Exception {
		String suffix = unique();
		mvc.perform(api.jsonPost("/register", registration("first_" + suffix, "same" + suffix + "@example.com", "password-123")))
			.andExpect(status().isCreated());
		mvc.perform(api.jsonPost("/register", registration("second_" + suffix, "SAME" + suffix + "@EXAMPLE.com", "password-123")))
			.andExpect(status().isConflict())
			.andExpect(jsonPath("$.code").value("EMAIL_TAKEN"));
	}

	@Test
	void invalidRegistrationIsABadRequestNamingTheFields() throws Exception {
		mvc.perform(api.jsonPost("/register", registration("x", "not-an-email", "short")))
			.andExpect(status().isBadRequest())
			.andExpect(jsonPath("$.code").value("VALIDATION_FAILED"))
			.andExpect(jsonPath("$.details.username").exists())
			.andExpect(jsonPath("$.details.email").exists())
			.andExpect(jsonPath("$.details.password").exists())
			.andExpect(jsonPath("$.requestId").isNotEmpty());
	}

	@Test
	void malformedJsonIsABadRequest() throws Exception {
		mvc.perform(from(post("/register").contentType(MediaType.APPLICATION_JSON).content("{\"username\":"),
				freshAddress()))
			.andExpect(status().isBadRequest())
			.andExpect(jsonPath("$.code").value("MALFORMED_REQUEST"));
	}

	@Test
	void loginReturnsAVerifiableTokenWithTheConfiguredExpiry() throws Exception {
		ApiTestSupport.Account account = api.registerAndLogin();
		var jwt = decoder.decode(account.token());
		assertThat(jwt.getSubject()).isEqualTo(account.userId());

		mvc.perform(api.jsonPost("/login", Map.of("email", account.email().toUpperCase(), "password", account.password())))
			.andExpect(status().isOk())
			.andExpect(header().string("Cache-Control", "no-store"))
			.andExpect(jsonPath("$.tokenType").value("Bearer"))
			.andExpect(jsonPath("$.expiresIn").value(3600))
			.andExpect(jsonPath("$.accessToken").isNotEmpty());
	}

	@Test
	void wrongPasswordAndUnknownEmailGetTheSameAnswer() throws Exception {
		ApiTestSupport.Account account = api.registerAndLogin();
		MvcResult wrong = mvc.perform(api.jsonPost("/login", Map.of("email", account.email(), "password", "wrong-password")))
			.andExpect(status().isUnauthorized())
			.andReturn();
		MvcResult unknown = mvc
			.perform(api.jsonPost("/login", Map.of("email", "nobody-" + unique() + "@example.com", "password", "wrong-password")))
			.andExpect(status().isUnauthorized())
			.andReturn();
		JsonNode a = api.body(wrong);
		JsonNode b = api.body(unknown);
		assertThat(a.get("code").asString()).isEqualTo("INVALID_CREDENTIALS").isEqualTo(b.get("code").asString());
		assertThat(a.get("message").asString()).isEqualTo(b.get("message").asString());
	}

	@Test
	void deactivatedAccountCannotLogIn() throws Exception {
		ApiTestSupport.Account account = api.registerAndLogin();
		jdbc.sql("update user_table set is_active = false where user_id = ?").param(UUID.fromString(account.userId())).update();
		mvc.perform(api.jsonPost("/login", Map.of("email", account.email(), "password", account.password())))
			.andExpect(status().isUnauthorized())
			.andExpect(jsonPath("$.code").value("INVALID_CREDENTIALS"));
	}

	@Test
	void repeatedLoginAttemptsAreRateLimited() throws Exception {
		String email = "target-" + unique() + "@example.com";
		int refused = 0;
		for (int i = 0; i < 12; i++) {
			int code = mvc.perform(api.jsonPost("/login", Map.of("email", email, "password", "guess-" + i)))
				.andReturn()
				.getResponse()
				.getStatus();
			if (code == 429) {
				refused++;
			}
		}
		assertThat(refused).isEqualTo(2);
		mvc.perform(api.jsonPost("/login", Map.of("email", email, "password", "guess")))
			.andExpect(status().isTooManyRequests())
			.andExpect(header().exists("Retry-After"))
			.andExpect(jsonPath("$.code").value("TOO_MANY_REQUESTS"));
	}

	@Test
	void protectedRoutesNeedAToken() throws Exception {
		mvc.perform(from(post("/query").contentType(MediaType.APPLICATION_JSON).content("{\"query\":\"hi\"}"),
				freshAddress()))
			.andExpect(status().isUnauthorized())
			.andExpect(header().string("WWW-Authenticate", "Bearer"))
			.andExpect(jsonPath("$.code").value("UNAUTHORIZED"));
		mvc.perform(from(post("/ingest"), freshAddress())).andExpect(status().isUnauthorized());
	}

	@Test
	void invalidTokensAreRejected() throws Exception {
		for (String token : new String[] { "garbage", "a.b.c",
				"eyJhbGciOiJub25lIn0.eyJzdWIiOiIxMjMifQ." }) {
			mvc.perform(from(post("/query").header("Authorization", "Bearer " + token)
				.contentType(MediaType.APPLICATION_JSON)
				.content("{\"query\":\"hi\"}"), freshAddress()))
				.andExpect(status().isUnauthorized())
				.andExpect(jsonPath("$.code").value("INVALID_TOKEN"));
		}
	}

	@Test
	void staleTokenOnLoginIsIgnored() throws Exception {
		ApiTestSupport.Account account = api.registerAndLogin();
		mvc.perform(api.jsonPost("/login", Map.of("email", account.email(), "password", account.password()))
			.header("Authorization", "Bearer expired.or.garbage"))
			.andExpect(status().isOk());
	}

	@Test
	void healthIsPublicAndDetailFree() throws Exception {
		mvc.perform(get("/actuator/health"))
			.andExpect(status().isOk())
			.andExpect(jsonPath("$.status").value("UP"))
			.andExpect(jsonPath("$.components").doesNotExist());
	}

	@Test
	void corsAllowsOnlyConfiguredOrigins() throws Exception {
		mvc.perform(from(org.springframework.test.web.servlet.request.MockMvcRequestBuilders.options("/login")
			.header("Origin", "http://localhost:5173")
			.header("Access-Control-Request-Method", "POST"), freshAddress()))
			.andExpect(status().isOk())
			.andExpect(header().string("Access-Control-Allow-Origin", "http://localhost:5173"))
			.andExpect(header().doesNotExist("Access-Control-Allow-Credentials"));
		mvc.perform(from(org.springframework.test.web.servlet.request.MockMvcRequestBuilders.options("/login")
			.header("Origin", "https://evil.example")
			.header("Access-Control-Request-Method", "POST"), freshAddress()))
			.andExpect(status().isForbidden());
	}

	@Test
	void databaseEnforcesUniquenessAndNormalisationItself() {
		UUID id = UUID.randomUUID();
		String suffix = unique();
		jdbc.sql("insert into user_table (user_id, username, email, password_hash) values (?, ?, ?, ?)")
			.params(id, "db_" + suffix, "db" + suffix + "@example.com", "$argon2id$v=19$x")
			.update();
		org.assertj.core.api.Assertions
			.assertThatThrownBy(() -> jdbc
				.sql("insert into user_table (user_id, username, email, password_hash) values (?, ?, ?, ?)")
				.params(UUID.randomUUID(), "DB_" + suffix, "other" + suffix + "@example.com", "$argon2id$v=19$x")
				.update())
			.isInstanceOf(DuplicateKeyException.class);
		org.assertj.core.api.Assertions
			.assertThatThrownBy(() -> jdbc
				.sql("insert into user_table (user_id, username, email, password_hash) values (?, ?, ?, ?)")
				.params(UUID.randomUUID(), "mixed_" + suffix, "Mixed" + suffix + "@example.com", "$argon2id$v=19$x")
				.update())
			.hasMessageContaining("user_table_email_normalised");
		org.assertj.core.api.Assertions
			.assertThatThrownBy(() -> jdbc
				.sql("insert into user_table (user_id, username, email, password_hash) values (?, ?, ?, ?)")
				.params(UUID.randomUUID(), "plain_" + suffix, "plain" + suffix + "@example.com", "plaintext")
				.update())
			.hasMessageContaining("user_table_password_hash_encoded");
	}
}
