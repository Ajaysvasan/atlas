package com.fyp.atlas.grpc;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Duration;
import java.util.List;
import java.util.Set;
import java.util.UUID;

import com.fyp.atlas.config.AtlasProperties;
import com.fyp.atlas.dto.QueryResponseDto;
import com.fyp.atlas.exception.ApiException;
import com.fyp.atlas.exception.ErrorCode;
import com.fyp.atlas.grpc.v1.QueryRequest;
import com.fyp.atlas.grpc.v1.QueryResponse;
import com.fyp.atlas.grpc.v1.QueryStatus;
import com.fyp.atlas.service.QResponseHandler;
import io.grpc.ManagedChannel;
import io.grpc.Metadata;
import io.grpc.Status;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.util.unit.DataSize;

class GrpcBoundaryTest {

	static ApiException translated(Status status, String code) {
		Metadata trailers = new Metadata();
		if (code != null) {
			trailers.put(GrpcErrorTranslator.ERROR_CODE, code);
		}
		return GrpcErrorTranslator.translate(status.withDescription("/srv/app/x.py line 9").asRuntimeException(trailers));
	}

	@ParameterizedTest
	@CsvSource({ "NOT_FOUND,CONVERSATION_NOT_FOUND,CONVERSATION_NOT_FOUND", "INVALID_ARGUMENT,QUERY_TOO_LARGE,VALIDATION_FAILED",
			"INVALID_ARGUMENT,INVALID_REQUEST,VALIDATION_FAILED", "RESOURCE_EXHAUSTED,BUSY,BACKEND_BUSY",
			"RESOURCE_EXHAUSTED,TOO_LARGE,PAYLOAD_TOO_LARGE", "DEADLINE_EXCEEDED,,BACKEND_TIMEOUT",
			"UNAVAILABLE,MODEL_UNAVAILABLE,BACKEND_UNAVAILABLE", "UNAVAILABLE,DATABASE_UNAVAILABLE,BACKEND_UNAVAILABLE",
			"CANCELLED,,BACKEND_UNAVAILABLE", "UNAUTHENTICATED,,BACKEND_UNAVAILABLE", "INTERNAL,INTERNAL,INTERNAL_ERROR",
			"UNKNOWN,,INTERNAL_ERROR", "DATA_LOSS,,INTERNAL_ERROR" })
	void everyStatusHasAnHttpMeaning(String status, String code, ErrorCode expected) {
		ApiException error = translated(Status.fromCode(Status.Code.valueOf(status)), code);
		assertThat(error.code()).isEqualTo(expected);
		assertThat(error.getMessage()).doesNotContain("/srv").doesNotContain("line 9");
	}

	@Test
	void busyTellsTheClientWhenToRetry() {
		assertThat(translated(Status.RESOURCE_EXHAUSTED, "BUSY").retryAfter()).isPositive();
	}

	static QueryResponse.Builder answer() {
		return QueryResponse.newBuilder()
			.setRequestId("r-12345678")
			.setConversationId("c1")
			.setConversationName("MVCC")
			.setAnswer("text")
			.setStatus(QueryStatus.QUERY_STATUS_COMPLETED_WITH_WARNINGS)
			.setExecutionTimeMs(42)
			.setRetryCount(2)
			.addWarnings("not saved")
			.setPersisted(false);
	}

	@Test
	void qResponseMapsEveryField() {
		QueryResponseDto dto = new QResponseHandler().handle("r-12345678", answer().build());
		assertThat(dto).isEqualTo(new QueryResponseDto("r-12345678", "c1", "MVCC", false, "text",
				"COMPLETED_WITH_WARNINGS", 42, 2, List.of("not saved"), false));
	}

	@Test
	void qResponseRefusesMismatchedIncompleteOrUnknownAnswers() {
		QResponseHandler handler = new QResponseHandler();
		assertThatThrownBy(() -> handler.handle("other-request", answer().build())).isInstanceOf(ApiException.class);
		assertThatThrownBy(() -> handler.handle("r-12345678", answer().setAnswer("").build()))
			.isInstanceOf(ApiException.class);
		assertThatThrownBy(() -> handler.handle("r-12345678", answer().setConversationId("").build()))
			.isInstanceOf(ApiException.class);
		assertThatThrownBy(() -> handler.handle("r-12345678",
				answer().setStatus(QueryStatus.QUERY_STATUS_UNSPECIFIED).build()))
			.isInstanceOfSatisfying(ApiException.class, e -> assertThat(e.code()).isEqualTo(ErrorCode.INTERNAL_ERROR));
	}

	static AtlasProperties properties(int port) {
		AtlasProperties.Grpc grpc = new AtlasProperties.Grpc("127.0.0.1", port, "a-service-token-long-enough",
				Duration.ofSeconds(2), Duration.ofSeconds(2), Duration.ofSeconds(2), Duration.ofSeconds(2),
				DataSize.ofMegabytes(4),
				DataSize.ofKilobytes(64), new AtlasProperties.Tls(false, null, null, null, null));
		return new AtlasProperties(null, null, null, null, grpc, new AtlasProperties.Upload(10,
				DataSize.ofMegabytes(25), DataSize.ofMegabytes(100), Set.of("txt"), DataSize.ofMegabytes(200), 10, 100),
				null);
	}

	@Test
	void anUnreachablePipelineIsServiceUnavailableNotACrash() throws Exception {
		ManagedChannel channel = NettyChannelBuilder.forAddress("127.0.0.1", 1).usePlaintext().build();
		try {
			AtlasGrpcClient client = new AtlasGrpcClient(channel, properties(1));
			assertThatThrownBy(() -> client.query(QueryRequest.newBuilder().setRequestId("r").setUserId("u").setQuery("q").build()))
				.isInstanceOfSatisfying(ApiException.class, e -> assertThat(e.code()).isEqualTo(ErrorCode.BACKEND_UNAVAILABLE));
			assertThatThrownBy(() -> client.ingest("r", UUID.randomUUID(), List.of(new AtlasGrpcClient.UploadSource(
					"a.txt", "text/plain", 1, () -> new java.io.ByteArrayInputStream(new byte[] { 'a' })))))
				.isInstanceOfSatisfying(ApiException.class, e -> assertThat(e.code()).isEqualTo(ErrorCode.BACKEND_UNAVAILABLE));
		}
		finally {
			channel.shutdownNow();
		}
	}

	@Test
	void theChannelRefusesToStartWithoutAServiceToken() {
		AtlasProperties.Grpc grpc = properties(1).grpc();
		AtlasProperties noToken = new AtlasProperties(null, null, null, null,
				new AtlasProperties.Grpc(grpc.host(), grpc.port(), "short", grpc.queryDeadline(), grpc.ingestDeadline(),
						grpc.healthDeadline(), grpc.listDeadline(), grpc.maxInboundMessageSize(), grpc.chunkSize(),
						grpc.tls()),
				null, null);
		assertThatThrownBy(() -> new GrpcChannelConfig().atlasChannel(noToken))
			.hasMessageContaining("ATLAS_GRPC_SERVICE_TOKEN");
	}
}
