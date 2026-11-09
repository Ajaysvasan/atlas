package com.fyp.atlas.exception;

import java.io.IOException;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import com.fyp.atlas.web.RequestCorrelation;
import tools.jackson.databind.json.JsonMapper;

import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;

/** Builds the ApiError body, for MVC handlers and for the security filters that run before MVC. */
@Component
public class ApiErrorWriter {

	private final JsonMapper json;

	public ApiErrorWriter(JsonMapper json) {
		this.json = json;
	}

	public ApiError body(ErrorCode code, String message, HttpServletRequest request, Object details) {
		return ApiError.of(code, message, request.getRequestURI(), RequestCorrelation.of(request), details);
	}

	public ResponseEntity<ApiError> entity(ErrorCode code, String message, HttpServletRequest request,
			Object details) {
		return ResponseEntity.status(code.status())
			.contentType(MediaType.APPLICATION_JSON)
			.body(body(code, message, request, details));
	}

	public void write(HttpServletResponse response, ErrorCode code, HttpServletRequest request) throws IOException {
		response.setStatus(code.status().value());
		response.setContentType(MediaType.APPLICATION_JSON_VALUE);
		response.setCharacterEncoding("UTF-8");
		json.writeValue(response.getOutputStream(), body(code, code.defaultMessage(), request, null));
	}
}
