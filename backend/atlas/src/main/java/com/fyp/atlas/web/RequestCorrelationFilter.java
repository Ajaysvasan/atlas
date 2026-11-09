package com.fyp.atlas.web;

import java.io.IOException;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

/** Gives every request an id, echoes it back, and puts it on every log line; runs before security. */
@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
public class RequestCorrelationFilter extends OncePerRequestFilter {

	private static final Logger log = LoggerFactory.getLogger(RequestCorrelationFilter.class);

	@Override
	protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain chain)
			throws ServletException, IOException {
		String id = RequestCorrelation.accept(request.getHeader(RequestCorrelation.HEADER));
		request.setAttribute(RequestCorrelation.ATTRIBUTE, id);
		response.setHeader(RequestCorrelation.HEADER, id);
		MDC.put(RequestCorrelation.MDC_KEY, id);
		long started = System.nanoTime();
		try {
			chain.doFilter(request, response);
		}
		finally {
			log.info("{} {} -> {} in {} ms", request.getMethod(), request.getRequestURI(), response.getStatus(),
					(System.nanoTime() - started) / 1_000_000);
			MDC.remove(RequestCorrelation.MDC_KEY);
		}
	}
}
