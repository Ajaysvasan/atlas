package com.fyp.atlas.dto;

import java.util.List;

/** The answer to one query, with the pipeline's own execution metadata. */
public record QueryResponseDto(
		String requestId,
		String conversationId,
		String conversationName,
		boolean newConversation,
		String answer,
		String status,
		long executionTimeMs,
		int retryCount,
		List<String> warnings,
		boolean persisted) {
}
