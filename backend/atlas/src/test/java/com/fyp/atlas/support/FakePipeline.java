package com.fyp.atlas.support;

import java.io.ByteArrayOutputStream;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

import com.fyp.atlas.grpc.v1.AtlasServiceGrpc;
import com.fyp.atlas.grpc.v1.FileResult;
import com.fyp.atlas.grpc.v1.FileStatus;
import com.fyp.atlas.grpc.v1.HealthRequest;
import com.fyp.atlas.grpc.v1.HealthResponse;
import com.fyp.atlas.grpc.v1.IngestRequest;
import com.fyp.atlas.grpc.v1.IngestResponse;
import com.fyp.atlas.grpc.v1.IngestStatus;
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

	public final List<QueryRequest> queries = new CopyOnWriteArrayList<>();
	public final List<ReceivedUpload> uploads = new CopyOnWriteArrayList<>();
	public final Map<String, String> owners = new ConcurrentHashMap<>();
	public volatile Status failWith;
	public volatile String failCode;
	public volatile long stallMillis;
	public volatile FileStatus ingestStatus = FileStatus.FILE_STATUS_INGESTED;
	public volatile String ingestErrorCode = "";

	public void reset() {
		queries.clear();
		uploads.clear();
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
