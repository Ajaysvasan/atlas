package com.fyp.atlas.exception;

import java.util.LinkedHashMap;
import java.util.Map;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.ConstraintViolationException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.validation.FieldError;
import org.springframework.web.HttpMediaTypeNotSupportedException;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.multipart.MaxUploadSizeExceededException;
import org.springframework.web.multipart.MultipartException;
import org.springframework.web.multipart.support.MissingServletRequestPartException;
import org.springframework.web.servlet.resource.NoResourceFoundException;

/** Every exception becomes an ApiError; only ApiException messages are shown to the client verbatim. */
@RestControllerAdvice
public class GlobalExceptionHandler {

	private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

	private final ApiErrorWriter errors;

	public GlobalExceptionHandler(ApiErrorWriter errors) {
		this.errors = errors;
	}

	@ExceptionHandler(ApiException.class)
	ResponseEntity<ApiError> api(ApiException error, HttpServletRequest request) {
		if (error.code().status().is5xxServerError()) {
			log.warn("{}: {}", error.code(), error.getMessage(), error.getCause());
		}
		ResponseEntity<ApiError> response = errors.entity(error.code(), error.getMessage(), request, error.details());
		if (error.retryAfter() != null) {
			long seconds = Math.max(1, error.retryAfter().toSeconds());
			return ResponseEntity.status(response.getStatusCode())
				.headers(response.getHeaders())
				.header(HttpHeaders.RETRY_AFTER, Long.toString(seconds))
				.body(response.getBody());
		}
		return response;
	}

	@ExceptionHandler(MethodArgumentNotValidException.class)
	ResponseEntity<ApiError> invalid(MethodArgumentNotValidException error, HttpServletRequest request) {
		Map<String, String> fields = new LinkedHashMap<>();
		for (FieldError field : error.getBindingResult().getFieldErrors()) {
			fields.putIfAbsent(field.getField(), field.getDefaultMessage());
		}
		return errors.entity(ErrorCode.VALIDATION_FAILED, ErrorCode.VALIDATION_FAILED.defaultMessage(), request,
				fields);
	}

	@ExceptionHandler(ConstraintViolationException.class)
	ResponseEntity<ApiError> violated(ConstraintViolationException error, HttpServletRequest request) {
		Map<String, String> fields = new LinkedHashMap<>();
		error.getConstraintViolations()
			.forEach(v -> fields.putIfAbsent(v.getPropertyPath().toString(), v.getMessage()));
		return errors.entity(ErrorCode.VALIDATION_FAILED, ErrorCode.VALIDATION_FAILED.defaultMessage(), request,
				fields);
	}

	@ExceptionHandler(HttpMessageNotReadableException.class)
	ResponseEntity<ApiError> unreadable(HttpServletRequest request) {
		return errors.entity(ErrorCode.MALFORMED_REQUEST, ErrorCode.MALFORMED_REQUEST.defaultMessage(), request, null);
	}

	@ExceptionHandler(MaxUploadSizeExceededException.class)
	ResponseEntity<ApiError> tooLarge(HttpServletRequest request) {
		return errors.entity(ErrorCode.PAYLOAD_TOO_LARGE,
				"The upload exceeds the per-file or per-request size limit.", request, null);
	}

	@ExceptionHandler(MissingServletRequestPartException.class)
	ResponseEntity<ApiError> missingPart(MissingServletRequestPartException error, HttpServletRequest request) {
		return errors.entity(ErrorCode.VALIDATION_FAILED, "No files were uploaded.", request,
				Map.of(error.getRequestPartName(), "required"));
	}

	@ExceptionHandler(MultipartException.class)
	ResponseEntity<ApiError> multipart(MultipartException error, HttpServletRequest request) {
		Throwable cause = error;
		while (cause != null) {
			if (cause instanceof MaxUploadSizeExceededException
					|| cause.getClass().getSimpleName().contains("SizeLimitExceeded")) {
				return tooLarge(request);
			}
			cause = cause.getCause();
		}
		return errors.entity(ErrorCode.MALFORMED_REQUEST, "The multipart request could not be read.", request, null);
	}

	@ExceptionHandler(HttpMediaTypeNotSupportedException.class)
	ResponseEntity<ApiError> mediaType(HttpServletRequest request) {
		return errors.entity(ErrorCode.UNSUPPORTED_MEDIA_TYPE, "Unsupported request content type.", request, null);
	}

	@ExceptionHandler(HttpRequestMethodNotSupportedException.class)
	ResponseEntity<ApiError> method(HttpServletRequest request) {
		return errors.entity(ErrorCode.METHOD_NOT_ALLOWED, ErrorCode.METHOD_NOT_ALLOWED.defaultMessage(), request,
				null);
	}

	@ExceptionHandler(NoResourceFoundException.class)
	ResponseEntity<ApiError> notFound(HttpServletRequest request) {
		return errors.entity(ErrorCode.NOT_FOUND, ErrorCode.NOT_FOUND.defaultMessage(), request, null);
	}

	@ExceptionHandler(Exception.class)
	ResponseEntity<ApiError> unexpected(Exception error, HttpServletRequest request) {
		log.error("Unhandled error on {} {}", request.getMethod(), request.getRequestURI(), error);
		return errors.entity(ErrorCode.INTERNAL_ERROR, ErrorCode.INTERNAL_ERROR.defaultMessage(), request, null);
	}
}
