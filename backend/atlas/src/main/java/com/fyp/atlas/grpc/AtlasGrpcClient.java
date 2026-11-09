package com.fyp.atlas.grpc;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import com.fyp.atlas.config.AtlasProperties;
import com.fyp.atlas.exception.ApiException;
import com.fyp.atlas.exception.ErrorCode;
import com.fyp.atlas.grpc.v1.AtlasServiceGrpc;
import com.fyp.atlas.grpc.v1.FileChunk;
import com.fyp.atlas.grpc.v1.FileEnd;
import com.fyp.atlas.grpc.v1.FileStart;
import com.fyp.atlas.grpc.v1.HealthRequest;
import com.fyp.atlas.grpc.v1.HealthResponse;
import com.fyp.atlas.grpc.v1.IngestHeader;
import com.fyp.atlas.grpc.v1.IngestRequest;
import com.fyp.atlas.grpc.v1.IngestResponse;
import com.fyp.atlas.grpc.v1.ListConversationsRequest;
import com.fyp.atlas.grpc.v1.ListConversationsResponse;
import com.fyp.atlas.grpc.v1.ListExchangesRequest;
import com.fyp.atlas.grpc.v1.ListExchangesResponse;
import com.fyp.atlas.grpc.v1.QueryRequest;
import com.fyp.atlas.grpc.v1.QueryResponse;
import com.google.protobuf.ByteString;
import io.grpc.ManagedChannel;
import io.grpc.Metadata;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.ClientCallStreamObserver;
import io.grpc.stub.ClientResponseObserver;
import io.grpc.stub.MetadataUtils;

import org.springframework.stereotype.Component;

/**
 * Typed calls to the Python pipeline. Every call carries the service token and the request id, has a
 * deadline, and is never retried: a query or an ingestion may already have committed state.
 */
@Component
public class AtlasGrpcClient {

	static final Metadata.Key<String> AUTHORIZATION = Metadata.Key.of("authorization", Metadata.ASCII_STRING_MARSHALLER);
	static final Metadata.Key<String> REQUEST_ID = Metadata.Key.of("x-request-id", Metadata.ASCII_STRING_MARSHALLER);

	private final AtlasProperties.Grpc settings;
	private final AtlasServiceGrpc.AtlasServiceBlockingStub blocking;
	private final AtlasServiceGrpc.AtlasServiceStub async;

	public AtlasGrpcClient(ManagedChannel atlasChannel, AtlasProperties properties) {
		this.settings = properties.grpc();
		this.blocking = AtlasServiceGrpc.newBlockingStub(atlasChannel);
		this.async = AtlasServiceGrpc.newStub(atlasChannel);
	}

	/** One file to stream: opened only when its turn comes, so uploads are never all in memory at once. */
	public record UploadSource(String filename, String contentType, long size, Opener opener) {
	}

	@FunctionalInterface
	public interface Opener {
		InputStream open() throws IOException;
	}

	private Metadata headers(String requestId) {
		Metadata headers = new Metadata();
		headers.put(AUTHORIZATION, "Bearer " + settings.serviceToken());
		headers.put(REQUEST_ID, requestId);
		return headers;
	}

	public QueryResponse query(QueryRequest request) {
		try {
			return blocking.withInterceptors(MetadataUtils.newAttachHeadersInterceptor(headers(request.getRequestId())))
				.withDeadlineAfter(settings.queryDeadline().toMillis(), TimeUnit.MILLISECONDS)
				.query(request);
		}
		catch (StatusRuntimeException failure) {
			throw GrpcErrorTranslator.translate(failure);
		}
	}

	public HealthResponse health(String requestId) {
		try {
			return blocking.withInterceptors(MetadataUtils.newAttachHeadersInterceptor(headers(requestId)))
				.withDeadlineAfter(settings.healthDeadline().toMillis(), TimeUnit.MILLISECONDS)
				.health(HealthRequest.getDefaultInstance());
		}
		catch (StatusRuntimeException failure) {
			throw GrpcErrorTranslator.translate(failure);
		}
	}

	/** Read-only, so it never waits behind a running query; a short deadline of its own. */
	public ListConversationsResponse listConversations(ListConversationsRequest request) {
		try {
			return blocking.withInterceptors(MetadataUtils.newAttachHeadersInterceptor(headers(request.getRequestId())))
				.withDeadlineAfter(settings.listDeadline().toMillis(), TimeUnit.MILLISECONDS)
				.listConversations(request);
		}
		catch (StatusRuntimeException failure) {
			throw GrpcErrorTranslator.translate(failure);
		}
	}

	public ListExchangesResponse listExchanges(ListExchangesRequest request) {
		try {
			return blocking.withInterceptors(MetadataUtils.newAttachHeadersInterceptor(headers(request.getRequestId())))
				.withDeadlineAfter(settings.listDeadline().toMillis(), TimeUnit.MILLISECONDS)
				.listExchanges(request);
		}
		catch (StatusRuntimeException failure) {
			throw GrpcErrorTranslator.translate(failure);
		}
	}

	/** Streams the files in order, honouring gRPC flow control, and returns Python's per-file report. */
	public IngestResponse ingest(String requestId, UUID userId, List<UploadSource> files) {
		CompletableFuture<IngestResponse> result = new CompletableFuture<>();
		Object ready = new Object();
		ClientResponseObserver<IngestRequest, IngestResponse> responses = new ClientResponseObserver<>() {
			@Override
			public void beforeStart(ClientCallStreamObserver<IngestRequest> requests) {
				requests.setOnReadyHandler(() -> {
					synchronized (ready) {
						ready.notifyAll();
					}
				});
			}

			@Override
			public void onNext(IngestResponse value) {
				result.complete(value);
			}

			@Override
			public void onError(Throwable failure) {
				result.completeExceptionally(failure);
				synchronized (ready) {
					ready.notifyAll();
				}
			}

			@Override
			public void onCompleted() {
				result.completeExceptionally(Status.INTERNAL.withDescription("no response").asRuntimeException());
			}
		};
		long deadlineMillis = settings.ingestDeadline().toMillis();
		@SuppressWarnings("unchecked")
		ClientCallStreamObserver<IngestRequest> requests = (ClientCallStreamObserver<IngestRequest>) async
			.withInterceptors(MetadataUtils.newAttachHeadersInterceptor(headers(requestId)))
			.withDeadlineAfter(deadlineMillis, TimeUnit.MILLISECONDS)
			.ingest(responses);
		try {
			send(requests, ready, result, IngestRequest.newBuilder()
				.setHeader(IngestHeader.newBuilder()
					.setRequestId(requestId)
					.setUserId(userId.toString())
					.setFileCount(files.size()))
				.build());
			int chunkSize = (int) settings.chunkSize().toBytes();
			for (int index = 0; index < files.size() && !result.isDone(); index++) {
				streamFile(requests, ready, result, index, files.get(index), chunkSize);
			}
			if (!result.isDone()) {
				requests.onCompleted();
			}
		}
		catch (IOException | UncheckedIOException failure) {
			requests.cancel("reading the upload failed", failure);
			throw new ApiException(ErrorCode.MALFORMED_REQUEST, "An uploaded file could not be read.", null, null,
					failure);
		}
		catch (InterruptedException failure) {
			Thread.currentThread().interrupt();
			requests.cancel("interrupted", failure);
			throw new ApiException(ErrorCode.BACKEND_UNAVAILABLE);
		}
		catch (RuntimeException failure) {
			requests.cancel("upload aborted", failure);
			throw failure;
		}
		try {
			return result.get(deadlineMillis + 5_000, TimeUnit.MILLISECONDS);
		}
		catch (ExecutionException failure) {
			throw GrpcErrorTranslator.translate(failure.getCause());
		}
		catch (TimeoutException failure) {
			requests.cancel("no response before the deadline", failure);
			throw new ApiException(ErrorCode.BACKEND_TIMEOUT);
		}
		catch (InterruptedException failure) {
			Thread.currentThread().interrupt();
			requests.cancel("interrupted", failure);
			throw new ApiException(ErrorCode.BACKEND_UNAVAILABLE);
		}
	}

	private void streamFile(ClientCallStreamObserver<IngestRequest> requests, Object ready,
			CompletableFuture<IngestResponse> result, int index, UploadSource file, int chunkSize)
			throws IOException, InterruptedException {
		send(requests, ready, result, IngestRequest.newBuilder()
			.setFileStart(FileStart.newBuilder()
				.setFileIndex(index)
				.setFilename(file.filename())
				.setContentType(file.contentType() == null ? "" : file.contentType())
				.setSizeBytes(file.size()))
			.build());
		MessageDigest sha256 = sha256();
		byte[] buffer = new byte[chunkSize];
		try (InputStream in = file.opener().open()) {
			int read;
			while ((read = in.readNBytes(buffer, 0, buffer.length)) > 0 && !result.isDone()) {
				sha256.update(buffer, 0, read);
				send(requests, ready, result, IngestRequest.newBuilder()
					.setChunk(FileChunk.newBuilder().setFileIndex(index).setData(ByteString.copyFrom(buffer, 0, read)))
					.build());
			}
		}
		send(requests, ready, result, IngestRequest.newBuilder()
			.setFileEnd(FileEnd.newBuilder().setFileIndex(index).setSha256(HexFormat.of().formatHex(sha256.digest())))
			.build());
	}

	/** Waits for transport readiness, so a slow Python side never makes Java buffer the whole upload. */
	private static void send(ClientCallStreamObserver<IngestRequest> requests, Object ready,
			CompletableFuture<IngestResponse> result, IngestRequest message) throws InterruptedException {
		synchronized (ready) {
			while (!requests.isReady() && !result.isDone()) {
				ready.wait(100);
			}
		}
		if (!result.isDone()) {
			requests.onNext(message);
		}
	}

	private static MessageDigest sha256() {
		try {
			return MessageDigest.getInstance("SHA-256");
		}
		catch (NoSuchAlgorithmException impossible) {
			throw new IllegalStateException(impossible);
		}
	}
}
