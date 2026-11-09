package com.fyp.atlas.controller;

import static com.fyp.atlas.support.ApiTestSupport.freshAddress;
import static com.fyp.atlas.support.ApiTestSupport.from;
import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.UUID;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import com.fyp.atlas.grpc.v1.FileStatus;
import com.fyp.atlas.grpc.v1.QueryRequest;
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
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;
import org.springframework.test.web.servlet.request.MockMultipartHttpServletRequestBuilder;

/** /query and /ingest through Spring, real PostgreSQL and a real gRPC client, against the in-process pipeline contract. */
@SpringBootTest
@AutoConfigureMockMvc
@Import({ PostgresContainer.class, FakePipelineConfig.class })
class PipelineIntegrationTest {

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

	MockHttpServletRequestBuilder ask(ApiTestSupport.Account account, Object body) {
		return from(post("/query").header("Authorization", "Bearer " + account.token())
			.contentType(MediaType.APPLICATION_JSON)
			.content(json.writeValueAsString(body)), freshAddress());
	}

	MockMultipartHttpServletRequestBuilder upload(ApiTestSupport.Account account, MockMultipartFile... files) {
		var builder = multipart("/ingest");
		for (MockMultipartFile file : files) {
			builder.file(file);
		}
		return from(builder.header("Authorization", "Bearer " + account.token()), freshAddress());
	}

	static MockMultipartFile file(String name, byte[] data) {
		return new MockMultipartFile("files", name, "application/octet-stream", data);
	}

	static byte[] docx(byte[] body) throws Exception {
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		try (ZipOutputStream zip = new ZipOutputStream(out)) {
			zip.putNextEntry(new ZipEntry("[Content_Types].xml"));
			zip.write("<Types/>".getBytes());
			zip.putNextEntry(new ZipEntry("word/document.xml"));
			zip.write(body);
		}
		return out.toByteArray();
	}

	// ---------------------------------------------------------------- query

	@Test
	void aQueryRunsAsTheTokensUserAndReturnsThePipelinesAnswer() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		MvcResult result = mvc.perform(ask(alice, Map.of("query", "  Explain how PostgreSQL MVCC works.  "))
			.header("X-Request-Id", "req-0123456789"))
			.andExpect(status().isOk())
			.andExpect(header().string("X-Request-Id", "req-0123456789"))
			.andExpect(jsonPath("$.requestId").value("req-0123456789"))
			.andExpect(jsonPath("$.conversationName").value("PostgreSQL MVCC"))
			.andExpect(jsonPath("$.newConversation").value(true))
			.andExpect(jsonPath("$.answer").value("PostgreSQL uses multiversion concurrency control."))
			.andExpect(jsonPath("$.status").value("COMPLETED"))
			.andExpect(jsonPath("$.executionTimeMs").value(1250))
			.andExpect(jsonPath("$.retryCount").value(1))
			.andReturn();
		QueryRequest sent = fake.queries.get(0);
		assertThat(sent.getUserId()).isEqualTo(alice.userId());
		assertThat(sent.getRequestId()).isEqualTo("req-0123456789");
		assertThat(sent.getQuery()).isEqualTo("Explain how PostgreSQL MVCC works.");
		assertThat(sent.hasConversationId()).isFalse();
		assertThat(api.body(result).get("conversationId").asString()).isNotBlank();
	}

	@Test
	void aUserIdInTheBodyCannotImpersonateAnotherUser() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		ApiTestSupport.Account bob = api.registerAndLogin();
		mvc.perform(ask(alice, Map.of("query", "hello", "userId", bob.userId(), "user_id", bob.userId())))
			.andExpect(status().isOk());
		assertThat(fake.queries.get(0).getUserId()).isEqualTo(alice.userId());
	}

	@Test
	void theOwnerContinuesTheirConversationAndNobodyElseCan() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		ApiTestSupport.Account bob = api.registerAndLogin();
		JsonNode first = api.body(mvc.perform(ask(alice, Map.of("query", "What is MVCC?"))).andReturn());
		String conversation = first.get("conversationId").asString();

		mvc.perform(ask(alice, Map.of("conversationId", conversation, "query", "And vacuum?")))
			.andExpect(status().isOk())
			.andExpect(jsonPath("$.conversationId").value(conversation))
			.andExpect(jsonPath("$.newConversation").value(false));
		mvc.perform(ask(bob, Map.of("conversationId", conversation, "query", "Show me alice's chat")))
			.andExpect(status().isNotFound())
			.andExpect(jsonPath("$.code").value("CONVERSATION_NOT_FOUND"));
		assertThat(fake.queries.get(2).getUserId()).isEqualTo(bob.userId());
	}

	@Test
	void invalidQueriesNeverReachThePipeline() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		mvc.perform(ask(alice, Map.of("query", "   "))).andExpect(status().isBadRequest())
			.andExpect(jsonPath("$.details.query").exists());
		mvc.perform(ask(alice, Map.of("query", "x".repeat(8001)))).andExpect(status().isBadRequest());
		mvc.perform(ask(alice, Map.of("query", "hi", "conversationId", "../../etc"))).andExpect(status().isBadRequest())
			.andExpect(jsonPath("$.details.conversationId").exists());
		assertThat(fake.queries).isEmpty();
	}

	@Test
	void pipelineFailuresBecomeTheRightHttpErrorsWithoutPythonDetails() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		Object[][] cases = { { Status.UNAVAILABLE, "MODEL_UNAVAILABLE", 503, "BACKEND_UNAVAILABLE" },
				{ Status.RESOURCE_EXHAUSTED, "BUSY", 503, "BACKEND_BUSY" },
				{ Status.INTERNAL, "INTERNAL", 500, "INTERNAL_ERROR" },
				{ Status.INVALID_ARGUMENT, "QUERY_TOO_LARGE", 400, "VALIDATION_FAILED" },
				{ Status.UNAUTHENTICATED, "UNAUTHENTICATED", 503, "BACKEND_UNAVAILABLE" } };
		for (Object[] c : cases) {
			fake.failWith = (Status) c[0];
			fake.failCode = (String) c[1];
			MvcResult result = mvc.perform(ask(alice, Map.of("query", "hello")))
				.andExpect(status().is((int) c[2]))
				.andExpect(jsonPath("$.code").value((String) c[3]))
				.andReturn();
			assertThat(result.getResponse().getContentAsString()).doesNotContain("secret").doesNotContain("boom");
		}
		fake.failWith = Status.RESOURCE_EXHAUSTED;
		fake.failCode = "BUSY";
		mvc.perform(ask(alice, Map.of("query", "hello"))).andExpect(header().exists("Retry-After"));
	}

	@Test
	void aPipelineThatOverrunsItsDeadlineIsAGatewayTimeout() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		fake.stallMillis = 3_000;
		mvc.perform(ask(alice, Map.of("query", "slow"))).andExpect(status().isGatewayTimeout())
			.andExpect(jsonPath("$.code").value("BACKEND_TIMEOUT"));
	}

	@Test
	void aDeactivatedAccountsTokenStopsWorking() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		jdbc.sql("update user_table set is_active = false where user_id = ?").param(UUID.fromString(alice.userId())).update();
		mvc.perform(ask(alice, Map.of("query", "hello"))).andExpect(status().isUnauthorized());
		assertThat(fake.queries).isEmpty();
	}

	// --------------------------------------------------------------- ingest

	@Test
	void severalFilesAreStreamedIntactAsTheTokensUser() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		byte[] notes = "# MVCC\nPostgreSQL keeps row versions.".getBytes(StandardCharsets.UTF_8);
		byte[] big = new byte[200 * 1024];
		java.util.Arrays.fill(big, (byte) 'a');
		byte[] doc = docx("<w:t>hello</w:t>".getBytes());
		mvc.perform(upload(alice, file("notes.md", notes), file("big.txt", big), file("report.docx", doc),
				file("paper.pdf", "%PDF-1.7\n%%EOF".getBytes())))
			.andExpect(status().isOk())
			.andExpect(jsonPath("$.status").value("COMPLETED"))
			.andExpect(jsonPath("$.succeeded").value(4))
			.andExpect(jsonPath("$.files[0].filename").value("notes.md"))
			.andExpect(jsonPath("$.files[0].documentId").isNotEmpty())
			.andExpect(jsonPath("$.storageLimitBytes").value(1L << 30));
		FakePipeline.ReceivedUpload received = fake.uploads.get(0);
		assertThat(received.userId()).isEqualTo(alice.userId());
		assertThat(received.files()).extracting(FakePipeline.ReceivedFile::filename)
			.containsExactly("notes.md", "big.txt", "report.docx", "paper.pdf");
		assertThat(received.files().get(1).data()).isEqualTo(big);
		assertThat(received.files().get(1).sha256()).isEqualTo(FakePipeline.sha256(big));
	}

	@Test
	void badFilesAreRejectedInJavaAndTheRestStillGoThrough() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		byte[] bomb = docx(new byte[3 * 1024 * 1024]);
		mvc.perform(upload(alice, file("ok.txt", "fine".getBytes()), file("tool.exe", new byte[] { 'M', 'Z' }),
				file("fake.pdf", "not a pdf".getBytes()), file("bin.txt", new byte[] { 'a', 0, 'b' }),
				file("bomb.docx", bomb), file("../../etc/passwd.md", "root".getBytes()),
				file("empty.txt", new byte[0])))
			.andExpect(status().isMultiStatus())
			.andExpect(jsonPath("$.status").value("PARTIAL"))
			.andExpect(jsonPath("$.succeeded").value(2))
			.andExpect(jsonPath("$.failed").value(5))
			.andExpect(jsonPath("$.files[1].errorCode").value("UNSUPPORTED_TYPE"))
			.andExpect(jsonPath("$.files[2].errorCode").value("MALFORMED"))
			.andExpect(jsonPath("$.files[3].errorCode").value("MALFORMED"))
			.andExpect(jsonPath("$.files[4].errorCode").value("TOO_LARGE"))
			.andExpect(jsonPath("$.files[5].filename").value("passwd.md"))
			.andExpect(jsonPath("$.files[5].status").value("INGESTED"))
			.andExpect(jsonPath("$.files[6].errorCode").value("EMPTY"));
		assertThat(fake.uploads.get(0).files()).extracting(FakePipeline.ReceivedFile::filename)
			.containsExactly("ok.txt", "passwd.md");
	}

	@Test
	void whenNothingCanBeIngestedTheStatusSaysWhy() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		mvc.perform(upload(alice, file("a.exe", new byte[] { 1 }), file("b.sh", "echo".getBytes())))
			.andExpect(status().isUnsupportedMediaType())
			.andExpect(jsonPath("$.status").value("FAILED"));
		mvc.perform(upload(alice, file("big.txt", new byte[300 * 1024])))
			.andExpect(status().isPayloadTooLarge());
		fake.ingestStatus = FileStatus.FILE_STATUS_NOT_INGESTED;
		fake.ingestErrorCode = "QUOTA_EXCEEDED";
		mvc.perform(upload(alice, file("a.txt", "text".getBytes())))
			.andExpect(status().isInsufficientStorage())
			.andExpect(jsonPath("$.files[0].status").value("NOT_INGESTED"))
			.andExpect(jsonPath("$.files[0].errorCode").value("QUOTA_EXCEEDED"));
		assertThat(fake.uploads).hasSize(1);
	}

	@Test
	void uploadLimitsAreEnforcedBeforeAnythingIsSent() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		MockMultipartFile[] eleven = new MockMultipartFile[11];
		for (int i = 0; i < 11; i++) {
			eleven[i] = file("f" + i + ".txt", "x".getBytes());
		}
		mvc.perform(upload(alice, eleven)).andExpect(status().isBadRequest());
		MockMultipartFile[] heavy = new MockMultipartFile[5];
		for (int i = 0; i < 5; i++) {
			heavy[i] = file("h" + i + ".txt", new byte[250 * 1024]);
		}
		mvc.perform(upload(alice, heavy)).andExpect(status().isPayloadTooLarge());
		mvc.perform(from(multipart("/ingest").header("Authorization", "Bearer " + alice.token()), freshAddress()))
			.andExpect(status().isBadRequest());
		assertThat(fake.uploads).isEmpty();
	}

	@Test
	void ingestionFailuresInThePipelineAreTranslated() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		fake.failWith = Status.UNAVAILABLE;
		mvc.perform(upload(alice, file("a.txt", "text".getBytes()))).andExpect(status().isServiceUnavailable());
		fake.failWith = null;
		fake.stallMillis = 4_000;
		mvc.perform(upload(alice, file("a.txt", "text".getBytes()))).andExpect(status().isGatewayTimeout());
	}

	@Test
	void eachUsersUploadIsSentUnderTheirOwnId() throws Exception {
		ApiTestSupport.Account alice = api.registerAndLogin();
		ApiTestSupport.Account bob = api.registerAndLogin();
		mvc.perform(upload(alice, file("a.txt", "alice's notes".getBytes()))).andExpect(status().isOk());
		mvc.perform(upload(bob, file("b.txt", "bob's notes".getBytes()))).andExpect(status().isOk());
		assertThat(fake.uploads).extracting(FakePipeline.ReceivedUpload::userId)
			.containsExactly(alice.userId(), bob.userId());
	}
}
