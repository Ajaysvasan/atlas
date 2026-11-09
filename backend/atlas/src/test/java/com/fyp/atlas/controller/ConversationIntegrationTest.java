package com.fyp.atlas.controller;

import static com.fyp.atlas.support.ApiTestSupport.freshAddress;
import static com.fyp.atlas.support.ApiTestSupport.from;
import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.time.OffsetDateTime;
import java.util.Map;
import java.util.UUID;

import com.fyp.atlas.grpc.v1.ListConversationsRequest;
import com.fyp.atlas.support.ApiTestSupport;
import com.fyp.atlas.support.FakePipeline;
import com.fyp.atlas.support.FakePipelineConfig;
import com.fyp.atlas.support.PostgresContainer;
import io.grpc.Status;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.context.annotation.Import;
import org.springframework.http.MediaType;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;

/** GET /conversations and /conversations/{id}/exchanges through Spring, real PostgreSQL and a real gRPC client. */
@SpringBootTest
@AutoConfigureMockMvc
@Import({ PostgresContainer.class, FakePipelineConfig.class })
class ConversationIntegrationTest {

	@Autowired
	MockMvc mvc;

	@Autowired
	JsonMapper json;

	@Autowired
	FakePipeline fake;

	@Autowired
	JdbcClient jdbc;

	ApiTestSupport api;

	@BeforeEach
	void setUp() {
		api = new ApiTestSupport(mvc, json);
		fake.reset();
	}

	MockHttpServletRequestBuilder list(ApiTestSupport.Account account, String query) {
		return from(get("/conversations" + query).header("Authorization", "Bearer " + account.token()), freshAddress());
	}

	MockHttpServletRequestBuilder exchanges(ApiTestSupport.Account account, String conversationId, String query) {
		return from(get("/conversations/" + conversationId + "/exchanges" + query)
			.header("Authorization", "Bearer " + account.token()), freshAddress());
	}

	String ask(ApiTestSupport.Account account, String conversationId, String query) throws Exception {
		Map<String, String> body = conversationId == null ? Map.of("query", query)
				: Map.of("conversationId", conversationId, "query", query);
		MvcResult result = mvc.perform(from(post("/query").header("Authorization", "Bearer " + account.token())
			.contentType(MediaType.APPLICATION_JSON)
			.content(json.writeValueAsString(body)), freshAddress()))
			.andExpect(status().isOk())
			.andReturn();
		return api.body(result).get("conversationId").asString();
	}

	// ---------------------------------------------------------------- list

	@Test
	void aTokenIsRequired() throws Exception {
		mvc.perform(get("/conversations")).andExpect(status().isUnauthorized());
		mvc.perform(get("/conversations/abc/exchanges")).andExpect(status().isUnauthorized());
		assertThat(fake.conversationListings).isEmpty();
		assertThat(fake.exchangeListings).isEmpty();
	}

	@Test
	void aNewUserHasNoConversations() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		mvc.perform(list(alice, "").header("X-Request-Id", "req-list-000001"))
			.andExpect(status().isOk())
			.andExpect(header().string("X-Request-Id", "req-list-000001"))
			.andExpect(jsonPath("$.requestId").value("req-list-000001"))
			.andExpect(jsonPath("$.conversations").isEmpty())
			.andExpect(jsonPath("$.nextCursor").doesNotExist());
		ListConversationsRequest sent = fake.conversationListings.get(0);
		assertThat(sent.getUserId()).isEqualTo(alice.userId());
		assertThat(sent.getLimit()).isEqualTo(50);
		assertThat(sent.getCursor()).isEmpty();
	}

	@Test
	void theCallersConversationsComeBackTypedAndNewestFirst() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		String first = ask(alice, null, "What is MVCC?");
		ask(alice, first, "And vacuum?");
		String second = ask(alice, null, "What is WAL?");

		JsonNode body = api.body(mvc.perform(list(alice, "")).andExpect(status().isOk()).andReturn());
		JsonNode rows = body.get("conversations");
		assertThat(rows.size()).isEqualTo(2);
		assertThat(rows.get(0).get("conversationId").asString()).isEqualTo(second);
		assertThat(rows.get(1).get("conversationId").asString()).isEqualTo(first);
		assertThat(rows.get(1).get("conversationName").asString()).isEqualTo("PostgreSQL MVCC");
		assertThat(rows.get(1).get("exchangeCount").asInt()).isEqualTo(2);
		OffsetDateTime created = OffsetDateTime.parse(rows.get(1).get("createdAt").asString());
		assertThat(OffsetDateTime.parse(rows.get(1).get("lastActivityAt").asString())).isAfterOrEqualTo(created);
	}

	@Test
	void aUserParameterCannotImpersonateAnotherUser() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		ApiTestSupport.Account bob = api.registerAndLogin();
		ask(bob, null, "Bob's question");
		mvc.perform(list(alice, "?userId=" + bob.userId() + "&user_id=" + bob.userId()))
			.andExpect(status().isOk())
			.andExpect(jsonPath("$.conversations").isEmpty());
		assertThat(fake.conversationListings.get(0).getUserId()).isEqualTo(alice.userId());
	}

	@Test
	void pagesFollowTheCursorToTheEnd() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		for (int n = 0; n < 3; n++) {
			ask(alice, null, "Question " + n);
		}
		JsonNode page1 = api.body(mvc.perform(list(alice, "?limit=2")).andExpect(status().isOk()).andReturn());
		assertThat(page1.get("conversations").size()).isEqualTo(2);
		String cursor = page1.get("nextCursor").asString();
		JsonNode page2 = api.body(mvc.perform(list(alice, "?limit=2&cursor=" + cursor)).andExpect(status().isOk())
			.andReturn());
		assertThat(page2.get("conversations").size()).isEqualTo(1);
		assertThat(page2.get("nextCursor").isNull()).isTrue();
		assertThat(fake.conversationListings.get(1).getCursor()).isEqualTo(cursor);
	}

	@Test
	void badPagingNeverReachesThePipeline() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		for (String limit : new String[] { "0", "201", "-1", "abc", "1.5" }) {
			mvc.perform(list(alice, "?limit=" + limit)).andExpect(status().isBadRequest())
				.andExpect(jsonPath("$.code").value("VALIDATION_FAILED"))
				.andExpect(jsonPath("$.details.limit").exists());
		}
		mvc.perform(list(alice, "?cursor=not%2Fa%2Bcursor")).andExpect(status().isBadRequest())
			.andExpect(jsonPath("$.details.cursor").exists());
		mvc.perform(list(alice, "?cursor=" + "A".repeat(513))).andExpect(status().isBadRequest())
			.andExpect(jsonPath("$.details.cursor").exists());
		assertThat(fake.conversationListings).isEmpty();
	}

	@Test
	void aCursorThePipelineRejectsIsABadRequest() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		mvc.perform(list(alice, "?cursor=bm90LW91cnM")).andExpect(status().isBadRequest())
			.andExpect(jsonPath("$.code").value("VALIDATION_FAILED"))
			.andExpect(jsonPath("$.details.cursor").exists());
	}

	@Test
	void pipelineFailuresBecomeTheRightHttpErrorsWithoutPythonDetails() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		Object[][] cases = { { Status.UNAVAILABLE, "DATABASE_UNAVAILABLE", 503, "BACKEND_UNAVAILABLE" },
				{ Status.INTERNAL, "INTERNAL", 500, "INTERNAL_ERROR" },
				{ Status.UNAUTHENTICATED, "UNAUTHENTICATED", 503, "BACKEND_UNAVAILABLE" } };
		for (Object[] c : cases) {
			fake.failWith = (Status) c[0];
			fake.failCode = (String) c[1];
			MvcResult result = mvc.perform(list(alice, ""))
				.andExpect(status().is((int) c[2]))
				.andExpect(jsonPath("$.code").value((String) c[3]))
				.andReturn();
			assertThat(result.getResponse().getContentAsString()).doesNotContain("secret").doesNotContain("boom");
		}
	}

	@Test
	void aListingThatOverrunsItsDeadlineIsAGatewayTimeout() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		fake.stallMillis = 3_000;
		mvc.perform(list(alice, "")).andExpect(status().isGatewayTimeout())
			.andExpect(jsonPath("$.code").value("BACKEND_TIMEOUT"));
	}

	@Test
	void anUnreadableTimestampFromThePipelineIsAnInternalError() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		ask(alice, null, "What is MVCC?");
		fake.timestampOverride = "yesterday";
		mvc.perform(list(alice, "")).andExpect(status().isInternalServerError())
			.andExpect(jsonPath("$.code").value("INTERNAL_ERROR"));
	}

	@Test
	void aDeactivatedAccountsTokenStopsWorking() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		jdbc.sql("update user_table set is_active = false where user_id = ?").param(UUID.fromString(alice.userId())).update();
		mvc.perform(list(alice, "")).andExpect(status().isUnauthorized());
		assertThat(fake.conversationListings).isEmpty();
	}

	// ----------------------------------------------------------- exchanges

	@Test
	void theOwnerReadsTheirExchangesOldestFirst() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		String conversation = ask(alice, null, "What is MVCC?");
		ask(alice, conversation, "And vacuum?");

		mvc.perform(exchanges(alice, conversation, ""))
			.andExpect(status().isOk())
			.andExpect(jsonPath("$.conversationId").value(conversation))
			.andExpect(jsonPath("$.conversationName").value("PostgreSQL MVCC"))
			.andExpect(jsonPath("$.exchanges.length()").value(2))
			.andExpect(jsonPath("$.exchanges[0].query").value("What is MVCC?"))
			.andExpect(jsonPath("$.exchanges[1].query").value("And vacuum?"))
			.andExpect(jsonPath("$.exchanges[0].answer").value("PostgreSQL uses multiversion concurrency control."))
			.andExpect(jsonPath("$.exchanges[0].askedAt").exists())
			.andExpect(jsonPath("$.exchanges[0].answeredAt").exists())
			.andExpect(jsonPath("$.nextCursor").doesNotExist());
		assertThat(fake.exchangeListings.get(0).getUserId()).isEqualTo(alice.userId());
	}

	@Test
	void anotherUsersConversationDoesNotExistForThem() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		ApiTestSupport.Account bob = api.registerAndLogin();
		String conversation = ask(alice, null, "Private question");
		MvcResult result = mvc.perform(exchanges(bob, conversation, ""))
			.andExpect(status().isNotFound())
			.andExpect(jsonPath("$.code").value("CONVERSATION_NOT_FOUND"))
			.andReturn();
		assertThat(result.getResponse().getContentAsString()).doesNotContain("Private question");
	}

	@Test
	void aMalformedIdIsNotFoundWithoutReachingThePipeline() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		mvc.perform(exchanges(alice, "-starts-with-a-dash", "")).andExpect(status().isNotFound())
			.andExpect(jsonPath("$.code").value("CONVERSATION_NOT_FOUND"));
		mvc.perform(exchanges(alice, "x".repeat(129), "")).andExpect(status().isNotFound());
		assertThat(fake.exchangeListings).isEmpty();
	}

	@Test
	void exchangePagesFollowTheCursor() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		String conversation = ask(alice, null, "q0");
		ask(alice, conversation, "q1");
		ask(alice, conversation, "q2");
		JsonNode page1 = api.body(mvc.perform(exchanges(alice, conversation, "?limit=2")).andReturn());
		assertThat(page1.get("exchanges").size()).isEqualTo(2);
		JsonNode page2 = api.body(mvc.perform(exchanges(alice, conversation,
				"?limit=2&cursor=" + page1.get("nextCursor").asString())).andReturn());
		assertThat(page2.get("exchanges").get(0).get("query").asString()).isEqualTo("q2");
		assertThat(page2.get("nextCursor").isNull()).isTrue();
	}
}
