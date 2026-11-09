package com.fyp.atlas.exception;

import java.time.Duration;
import java.util.Map;

/** An error with a client-safe message; anything else becomes INTERNAL_ERROR. */
public class ApiException extends RuntimeException {

	private final ErrorCode code;
	private final transient Object details;
	private final Duration retryAfter;

	public ApiException(ErrorCode code) {
		this(code, code.defaultMessage(), null, null, null);
	}

	public ApiException(ErrorCode code, String message) {
		this(code, message, null, null, null);
	}

	public ApiException(ErrorCode code, String message, Object details) {
		this(code, message, details, null, null);
	}

	public ApiException(ErrorCode code, String message, Object details, Duration retryAfter, Throwable cause) {
		super(message, cause);
		this.code = code;
		this.details = details;
		this.retryAfter = retryAfter;
	}

	public static ApiException tooManyRequests(Duration retryAfter) {
		return new ApiException(ErrorCode.TOO_MANY_REQUESTS, ErrorCode.TOO_MANY_REQUESTS.defaultMessage(), null,
				retryAfter, null);
	}

	public static ApiException validation(Map<String, String> fields) {
		return new ApiException(ErrorCode.VALIDATION_FAILED, ErrorCode.VALIDATION_FAILED.defaultMessage(), fields);
	}

	public ErrorCode code() {
		return code;
	}

	public Object details() {
		return details;
	}

	public Duration retryAfter() {
		return retryAfter;
	}
}
