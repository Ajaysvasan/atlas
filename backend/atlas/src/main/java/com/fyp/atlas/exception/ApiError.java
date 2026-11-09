package com.fyp.atlas.exception;

import java.time.Instant;

import com.fasterxml.jackson.annotation.JsonInclude;

/** The one error body every endpoint returns. Never carries stack traces, SQL or internal paths. */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record ApiError(
		Instant timestamp,
		int status,
		String error,
		String code,
		String message,
		String path,
		String requestId,
		Object details) {

	public static ApiError of(ErrorCode code, String message, String path, String requestId, Object details) {
		return new ApiError(Instant.now(), code.status().value(), code.status().getReasonPhrase(), code.name(),
				message, path, requestId, details);
	}
}
