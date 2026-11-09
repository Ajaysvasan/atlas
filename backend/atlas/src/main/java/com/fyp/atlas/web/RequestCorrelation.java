package com.fyp.atlas.web;

import java.util.UUID;
import java.util.regex.Pattern;

import jakarta.servlet.http.HttpServletRequest;

import org.slf4j.MDC;
import org.springframework.web.context.request.RequestAttributes;
import org.springframework.web.context.request.RequestContextHolder;

/** The request id: the caller's X-Request-Id when it is well formed, otherwise a new UUID. */
public final class RequestCorrelation {

	public static final String HEADER = "X-Request-Id";
	public static final String ATTRIBUTE = RequestCorrelation.class.getName() + ".id";
	public static final String MDC_KEY = "requestId";

	private static final Pattern SAFE = Pattern.compile("[A-Za-z0-9._-]{8,64}");

	private RequestCorrelation() {
	}

	static String accept(String candidate) {
		return candidate != null && SAFE.matcher(candidate).matches() ? candidate : UUID.randomUUID().toString();
	}

	public static String of(HttpServletRequest request) {
		Object id = request.getAttribute(ATTRIBUTE);
		return id != null ? id.toString() : current();
	}

	/** The id of the request being served on this thread; a fresh one outside a request. */
	public static String current() {
		RequestAttributes attributes = RequestContextHolder.getRequestAttributes();
		if (attributes != null) {
			Object id = attributes.getAttribute(ATTRIBUTE, RequestAttributes.SCOPE_REQUEST);
			if (id != null) {
				return id.toString();
			}
		}
		String logged = MDC.get(MDC_KEY);
		return logged != null ? logged : UUID.randomUUID().toString();
	}
}
