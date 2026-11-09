package com.fyp.atlas.exception;

import org.springframework.http.HttpStatus;

/** Every error a client can see: a stable code, its HTTP status and a safe default message. */
public enum ErrorCode {
	VALIDATION_FAILED(HttpStatus.BAD_REQUEST, "The request is invalid."),
	MALFORMED_REQUEST(HttpStatus.BAD_REQUEST, "The request body could not be read."),
	USERNAME_TAKEN(HttpStatus.CONFLICT, "That username is already registered."),
	EMAIL_TAKEN(HttpStatus.CONFLICT, "That email is already registered."),
	INVALID_CREDENTIALS(HttpStatus.UNAUTHORIZED, "Invalid email or password."),
	UNAUTHORIZED(HttpStatus.UNAUTHORIZED, "Authentication is required."),
	INVALID_TOKEN(HttpStatus.UNAUTHORIZED, "The access token is invalid or has expired."),
	FORBIDDEN(HttpStatus.FORBIDDEN, "You do not have access to this resource."),
	CONVERSATION_NOT_FOUND(HttpStatus.NOT_FOUND, "No such conversation."),
	NOT_FOUND(HttpStatus.NOT_FOUND, "Not found."),
	METHOD_NOT_ALLOWED(HttpStatus.METHOD_NOT_ALLOWED, "Method not allowed."),
	PAYLOAD_TOO_LARGE(HttpStatus.CONTENT_TOO_LARGE, "The upload is too large."),
	UNSUPPORTED_MEDIA_TYPE(HttpStatus.UNSUPPORTED_MEDIA_TYPE, "Unsupported file type."),
	UNPROCESSABLE_UPLOAD(HttpStatus.UNPROCESSABLE_CONTENT, "No file in the upload could be ingested."),
	QUOTA_EXCEEDED(HttpStatus.INSUFFICIENT_STORAGE, "The ingestion storage quota is exhausted."),
	TOO_MANY_REQUESTS(HttpStatus.TOO_MANY_REQUESTS, "Too many attempts; try again later."),
	BACKEND_BUSY(HttpStatus.SERVICE_UNAVAILABLE, "Atlas is busy; try again shortly."),
	BACKEND_UNAVAILABLE(HttpStatus.SERVICE_UNAVAILABLE, "The Atlas pipeline is unavailable."),
	BACKEND_TIMEOUT(HttpStatus.GATEWAY_TIMEOUT, "The Atlas pipeline did not answer in time."),
	INTERNAL_ERROR(HttpStatus.INTERNAL_SERVER_ERROR, "An unexpected error occurred.");

	private final HttpStatus status;
	private final String defaultMessage;

	ErrorCode(HttpStatus status, String defaultMessage) {
		this.status = status;
		this.defaultMessage = defaultMessage;
	}

	public HttpStatus status() {
		return status;
	}

	public String defaultMessage() {
		return defaultMessage;
	}
}
