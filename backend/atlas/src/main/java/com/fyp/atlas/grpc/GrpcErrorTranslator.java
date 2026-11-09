package com.fyp.atlas.grpc;

import java.time.Duration;

import com.fyp.atlas.exception.ApiException;
import com.fyp.atlas.exception.ErrorCode;
import io.grpc.Metadata;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A gRPC failure from the pipeline, as the REST error the client should see. Never forwards Python's text. */
public final class GrpcErrorTranslator {

	private static final Logger log = LoggerFactory.getLogger(GrpcErrorTranslator.class);

	public static final Metadata.Key<String> ERROR_CODE = Metadata.Key.of("atlas-error-code",
			Metadata.ASCII_STRING_MARSHALLER);

	private GrpcErrorTranslator() {
	}

	public static ApiException translate(Throwable failure) {
		Status status = Status.fromThrowable(failure);
		Metadata trailers = Status.trailersFromThrowable(failure);
		String code = trailers == null ? null : trailers.get(ERROR_CODE);
		log.warn("Pipeline call failed: {} {} ({})", status.getCode(), code == null ? "" : code,
				status.getDescription());
		return switch (status.getCode()) {
			case NOT_FOUND -> new ApiException(ErrorCode.CONVERSATION_NOT_FOUND);
			case INVALID_ARGUMENT -> "QUERY_TOO_LARGE".equals(code)
					? new ApiException(ErrorCode.VALIDATION_FAILED, "The query is too long.")
					: new ApiException(ErrorCode.VALIDATION_FAILED, "The pipeline could not process the request as given.");
			case RESOURCE_EXHAUSTED -> "TOO_LARGE".equals(code) ? new ApiException(ErrorCode.PAYLOAD_TOO_LARGE)
					: new ApiException(ErrorCode.BACKEND_BUSY, ErrorCode.BACKEND_BUSY.defaultMessage(), null,
							Duration.ofSeconds(30), failure);
			case DEADLINE_EXCEEDED -> new ApiException(ErrorCode.BACKEND_TIMEOUT);
			case UNAVAILABLE, CANCELLED, ABORTED -> new ApiException(ErrorCode.BACKEND_UNAVAILABLE,
					ErrorCode.BACKEND_UNAVAILABLE.defaultMessage(), null, null, failure);
			case UNAUTHENTICATED, PERMISSION_DENIED -> {
				// The service token is wrong: a deployment fault, not the user's.
				log.error("The Python pipeline refused this service's credentials; check ATLAS_GRPC_SERVICE_TOKEN");
				yield new ApiException(ErrorCode.BACKEND_UNAVAILABLE, ErrorCode.BACKEND_UNAVAILABLE.defaultMessage(),
						null, null, failure);
			}
			default -> new ApiException(ErrorCode.INTERNAL_ERROR, "The Atlas pipeline failed.", null, null, failure);
		};
	}

	public static boolean isGrpcFailure(Throwable failure) {
		return failure instanceof StatusRuntimeException || failure instanceof io.grpc.StatusException;
	}
}
