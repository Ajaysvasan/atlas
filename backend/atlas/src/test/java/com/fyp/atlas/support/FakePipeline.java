package com.fyp.atlas.support;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.Base64;
import java.util.Collections;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

import com.fyp.atlas.grpc.v1.AtlasServiceGrpc;
import com.fyp.atlas.grpc.v1.ConversationSummary;
import com.fyp.atlas.grpc.v1.ExchangeRecord;
import com.fyp.atlas.grpc.v1.FileResult;
import com.fyp.atlas.grpc.v1.FileStatus;
import com.fyp.atlas.grpc.v1.HealthRequest;
import com.fyp.atlas.grpc.v1.HealthResponse;
import com.fyp.atlas.grpc.v1.IngestRequest;
import com.fyp.atlas.grpc.v1.IngestResponse;
import com.fyp.atlas.grpc.v1.IngestStatus;
import com.fyp.atlas.grpc.v1.ListConversationsRequest;
import com.fyp.atlas.grpc.v1.ListConversationsResponse;
import com.fyp.atlas.grpc.v1.ListExchangesRequest;
import com.fyp.atlas.grpc.v1.ListExchangesResponse;
import com.fyp.atlas.grpc.v1.QueryRequest;
import com.fyp.atlas.grpc.v1.QueryResponse;
import com.fyp.atlas.grpc.v1.QueryStatus;
import io.grpc.Metadata;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;

/**
 * The Python service's contract, in-process: checks the service token, owns conversations per user
 * like the real registry, verifies streamed files, and can be told to fail or stall.
 */
public class FakePipeline extends AtlasServiceGrpc.AtlasServiceImplBase {

	public static final Metadata.Key<String> AUTHORIZATION = Metadata.Key.of("authorization",
			Metadata.ASCII_STRING_MARSHALLER);
	public static final Metadata.Key<String> ERROR_CODE = Metadata.Key.of("atlas-error-code",
			Metadata.ASCII_STRING_MARSHALLER);

	public record ReceivedFile(String filename, byte[] data, String sha256) {
	}

	public record ReceivedUpload(String requestId, String userId, List<ReceivedFile> files) {
	}

	public record FakeConversation(String id, String userId, String name, String createdAt,
			List<ExchangeRecord> exchanges) {
	}

	public final List<QueryRequest> queries = new CopyOnWriteArrayList<>();
	public final List<ReceivedUpload> uploads = new CopyOnWriteArrayList<>();
	public final Map<String, String> owners = new ConcurrentHashMap<>();
	/** Every conversation in creation order, with the exchanges /query recorded in it. */
	public final List<FakeConversation> conversations = new CopyOnWriteArrayList<>();
	public final List<ListConversationsRequest> conversationListings = new CopyOnWriteArrayList<>();
	public final List<ListExchangesRequest> exchangeListings = new CopyOnWriteArrayList<>();
	/** When set, every timestamp in a listing is replaced by this text. */
	public volatile String timestampOverride;
	public volatile Status failWith;
	public volatile String failCode;
	public volatile long stallMillis;
	public volatile FileStatus ingestStatus = FileStatus.FILE_STATUS_INGESTED;
	public volatile String ingestErrorCode = "";

	public void reset() {
		queries.clear();
		uploads.clear();
		conversationListings.clear();
		exchangeListings.clear();
		timestampOverride = null;
		failWith = null;
		failCode = null;
		stallMillis = 0;
		ingestStatus = FileStatus.FILE_STATUS_INGESTED;
		ingestErrorCode = "";
	}

	public ServerInterceptor tokenCheck(String token) {
		return new ServerInterceptor() {
			@Override
			public <Q, R> ServerCall.Listener<Q> interceptCall(ServerCall<Q, R> call, Metadata headers,
					ServerCallHandler<Q, R> next) {
				if (!("Bearer " + token).equals(headers.get(AUTHORIZATION))) {
					call.close(Status.UNAUTHENTICATED, new Metadata());
					return new ServerCall.Listener<>() {
					};
				}
				return next.startCall(call, headers);
			}
		};
	}

	private boolean failed(StreamObserver<?> observer) {
		if (failWith == null) {
			return false;
		}
		Metadata trailers = new Metadata();
		if (failCode != null) {
			trailers.put(ERROR_CODE, failCode);
		}
		observer.onError(failWith.withDescription("python said /secret/path.py:42 boom").asRuntimeException(trailers));
		return true;
	}

	private void stall() {
		if (stallMillis > 0) {
			try {
				Thread.sleep(stallMillis);
			}
			catch (InterruptedException error) {
				Thread.currentThread().interrupt();
			}
		}
	}

	@Override
	public void query(QueryRequest request, StreamObserver<QueryResponse> observer) {
		queries.add(request);
		stall();
		if (failed(observer)) {
			return;
		}
		String conversation;
		boolean created = !request.hasConversationId();
		if (created) {
			conversation = UUID.randomUUID().toString().replace("-", "");
			owners.put(conversation, request.getUserId());
		}
		else {
			conversation = request.getConversationId();
			if (!request.getUserId().equals(owners.get(conversation))) {
				Metadata trailers = new Metadata();
				trailers.put(ERROR_CODE, "CONVERSATION_NOT_FOUND");
				observer.onError(Status.NOT_FOUND.withDescription("no such conversation").asRuntimeException(trailers));
				return;
			}
		}
		String now = OffsetDateTime.now(ZoneOffset.UTC).toString();
		if (created) {
			conversations.add(new FakeConversation(conversation, request.getUserId(), "PostgreSQL MVCC", now,
					new CopyOnWriteArrayList<>()));
		}
		find(conversation).exchanges().add(ExchangeRecord.newBuilder()
			.setExchangeId(UUID.randomUUID().toString().replace("-", ""))
			.setQuery(request.getQuery())
			.setAskedAt(now)
			.setAnswer("PostgreSQL uses multiversion concurrency control.")
			.setAnsweredAt(now)
			.build());
		observer.onNext(QueryResponse.newBuilder()
			.setRequestId(request.getRequestId())
			.setConversationId(conversation)
			.setConversationName("PostgreSQL MVCC")
			.setNewConversation(created)
			.setAnswer("PostgreSQL uses multiversion concurrency control.")
			.setStatus(QueryStatus.QUERY_STATUS_COMPLETED)
			.setExecutionTimeMs(1250)
			.setRetryCount(1)
			.setPersisted(true)
			.build());
		observer.onCompleted();
	}

	@Override
	public StreamObserver<IngestRequest> ingest(StreamObserver<IngestResponse> observer) {
		return new StreamObserver<>() {
			String requestId;
			String userId;
			final Map<Integer, ByteArrayOutputStream> data = new LinkedHashMap<>();
			final Map<Integer, String> names = new LinkedHashMap<>();
			final Map<Integer, String> hashes = new LinkedHashMap<>();

			@Override
			public void onNext(IngestRequest message) {
				switch (message.getPartCase()) {
					case HEADER -> {
						requestId = message.getHeader().getRequestId();
						userId = message.getHeader().getUserId();
					}
					case FILE_START -> {
						names.put(message.getFileStart().getFileIndex(), message.getFileStart().getFilename());
						data.put(message.getFileStart().getFileIndex(), new ByteArrayOutputStream());
					}
					case CHUNK -> data.get(message.getChunk().getFileIndex()).writeBytes(message.getChunk().getData().toByteArray());
					case FILE_END -> hashes.put(message.getFileEnd().getFileIndex(), message.getFileEnd().getSha256());
					default -> {
					}
				}
			}

			@Override
			public void onError(Throwable error) {
			}

			@Override
			public void onCompleted() {
				stall();
				if (failed(observer)) {
					return;
				}
				List<ReceivedFile> files = new ArrayList<>();
				IngestResponse.Builder response = IngestResponse.newBuilder().setRequestId(requestId);
				int ok = 0;
				for (var entry : data.entrySet()) {
					byte[] bytes = entry.getValue().toByteArray();
					String sha = sha256(bytes);
					files.add(new ReceivedFile(names.get(entry.getKey()), bytes, hashes.get(entry.getKey())));
					boolean intact = sha.equals(hashes.get(entry.getKey()));
					FileStatus status = intact ? ingestStatus : FileStatus.FILE_STATUS_REJECTED;
					if (status == FileStatus.FILE_STATUS_INGESTED) {
						ok++;
					}
					response.addFiles(FileResult.newBuilder()
						.setFileIndex(entry.getKey())
						.setFilename(names.get(entry.getKey()))
						.setStatus(status)
						.setDocumentId(status == FileStatus.FILE_STATUS_INGESTED ? "doc-" + sha.substring(0, 8) : "")
						.setChunks(3)
						.setVectors(3)
						.setStoredBytes(4096)
						.setErrorCode(intact ? ingestErrorCode : "CHECKSUM_MISMATCH"));
				}
				uploads.add(new ReceivedUpload(requestId, userId, files));
				int failedCount = data.size() - ok;
				response.setSucceeded(ok)
					.setFailed(failedCount)
					.setStatus(failedCount == 0 ? IngestStatus.INGEST_STATUS_COMPLETED
							: ok == 0 ? IngestStatus.INGEST_STATUS_FAILED : IngestStatus.INGEST_STATUS_PARTIAL)
					.setStorageUsedBytes(10_000)
					.setStorageLimitBytes(1L << 30);
				observer.onNext(response.build());
				observer.onCompleted();
			}
		};
	}

	private FakeConversation find(String id) {
		return conversations.stream().filter(c -> c.id().equals(id)).findFirst().orElseThrow();
	}

	private String time(String value) {
		return timestampOverride == null ? value : timestampOverride;
	}

	/** The fake's cursor is an offset; the real one is a keyset. Callers treat both as opaque. */
	private static String cursorAt(int offset) {
		return Base64.getUrlEncoder().withoutPadding().encodeToString(("@" + offset).getBytes(StandardCharsets.UTF_8));
	}

	private static Integer offsetOf(String cursor) {
		if (cursor.isEmpty()) {
			return 0;
		}
		try {
			String text = new String(Base64.getUrlDecoder().decode(cursor), StandardCharsets.UTF_8);
			return text.startsWith("@") ? Integer.valueOf(text.substring(1)) : null;
		}
		catch (IllegalArgumentException malformed) {
			return null;
		}
	}

	private void invalidCursor(StreamObserver<?> observer) {
		Metadata trailers = new Metadata();
		trailers.put(ERROR_CODE, "INVALID_CURSOR");
		observer.onError(Status.INVALID_ARGUMENT.withDescription("the cursor is not valid").asRuntimeException(trailers));
	}

	@Override
	public void listConversations(ListConversationsRequest request, StreamObserver<ListConversationsResponse> observer) {
		conversationListings.add(request);
		stall();
		if (failed(observer)) {
			return;
		}
		Integer offset = offsetOf(request.getCursor());
		if (offset == null) {
			invalidCursor(observer);
			return;
		}
		List<FakeConversation> mine = new ArrayList<>(
				conversations.stream().filter(c -> c.userId().equals(request.getUserId())).toList());
		Collections.reverse(mine);
		int limit = request.getLimit() == 0 ? 50 : request.getLimit();
		int end = Math.min(mine.size(), offset + limit);
		ListConversationsResponse.Builder response = ListConversationsResponse.newBuilder()
			.setRequestId(request.getRequestId());
		for (FakeConversation c : mine.subList(Math.min(offset, end), end)) {
			response.addConversations(ConversationSummary.newBuilder()
				.setConversationId(c.id())
				.setConversationName(c.name())
				.setCreatedAt(time(c.createdAt()))
				.setLastActivityAt(time(c.exchanges().isEmpty() ? c.createdAt()
						: c.exchanges().get(c.exchanges().size() - 1).getAnsweredAt()))
				.setExchangeCount(c.exchanges().size()));
		}
		if (end < mine.size()) {
			response.setNextCursor(cursorAt(end));
		}
		observer.onNext(response.build());
		observer.onCompleted();
	}

	@Override
	public void listExchanges(ListExchangesRequest request, StreamObserver<ListExchangesResponse> observer) {
		exchangeListings.add(request);
		stall();
		if (failed(observer)) {
			return;
		}
		if (!request.getUserId().equals(owners.get(request.getConversationId()))) {
			Metadata trailers = new Metadata();
			trailers.put(ERROR_CODE, "CONVERSATION_NOT_FOUND");
			observer.onError(Status.NOT_FOUND.withDescription("no such conversation").asRuntimeException(trailers));
			return;
		}
		Integer offset = offsetOf(request.getCursor());
		if (offset == null) {
			invalidCursor(observer);
			return;
		}
		FakeConversation c = find(request.getConversationId());
		List<ExchangeRecord> all = c.exchanges();
		int limit = request.getLimit() == 0 ? 50 : request.getLimit();
		int end = Math.min(all.size(), offset + limit);
		ListExchangesResponse.Builder response = ListExchangesResponse.newBuilder()
			.setRequestId(request.getRequestId())
			.setConversationId(c.id())
			.setConversationName(c.name());
		for (ExchangeRecord e : all.subList(Math.min(offset, end), end)) {
			response.addExchanges(e.toBuilder().setAskedAt(time(e.getAskedAt())).setAnsweredAt(time(e.getAnsweredAt())));
		}
		if (end < all.size()) {
			response.setNextCursor(cursorAt(end));
		}
		observer.onNext(response.build());
		observer.onCompleted();
	}

	@Override
	public void health(HealthRequest request, StreamObserver<HealthResponse> observer) {
		observer.onNext(HealthResponse.newBuilder().setServing(true).setPipelineReady(true).build());
		observer.onCompleted();
	}

	public static String sha256(byte[] bytes) {
		try {
			return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
		}
		catch (Exception impossible) {
			throw new IllegalStateException(impossible);
		}
	}
}
