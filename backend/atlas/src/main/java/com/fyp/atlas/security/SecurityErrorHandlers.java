package com.fyp.atlas.security;

import java.io.IOException;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import com.fyp.atlas.exception.ApiErrorWriter;
import com.fyp.atlas.exception.ErrorCode;

import org.springframework.http.HttpHeaders;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.oauth2.server.resource.InvalidBearerTokenException;
import org.springframework.security.web.AuthenticationEntryPoint;
import org.springframework.security.web.access.AccessDeniedHandler;
import org.springframework.stereotype.Component;

/** 401 and 403 in the same ApiError shape as every other error, with the RFC 6750 challenge header. */
@Component
public class SecurityErrorHandlers implements AuthenticationEntryPoint, AccessDeniedHandler {

	private final ApiErrorWriter errors;

	public SecurityErrorHandlers(ApiErrorWriter errors) {
		this.errors = errors;
	}

	@Override
	public void commence(HttpServletRequest request, HttpServletResponse response,
			AuthenticationException exception) throws IOException {
		boolean presented = exception instanceof InvalidBearerTokenException
				|| exception.getClass().getSimpleName().contains("OAuth2");
		response.setHeader(HttpHeaders.WWW_AUTHENTICATE,
				presented ? "Bearer error=\"invalid_token\"" : "Bearer");
		errors.write(response, presented ? ErrorCode.INVALID_TOKEN : ErrorCode.UNAUTHORIZED, request);
	}

	@Override
	public void handle(HttpServletRequest request, HttpServletResponse response, AccessDeniedException exception)
			throws IOException {
		errors.write(response, ErrorCode.FORBIDDEN, request);
	}
}
