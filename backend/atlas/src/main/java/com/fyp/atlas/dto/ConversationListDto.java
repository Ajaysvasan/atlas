package com.fyp.atlas.dto;

import java.time.OffsetDateTime;
import java.util.List;

/** GET /conversations: one page of the caller's conversations, most recently active first. */
public record ConversationListDto(String requestId, List<Item> conversations, String nextCursor) {

	public record Item(
			String conversationId,
			String conversationName,
			OffsetDateTime createdAt,
			OffsetDateTime lastActivityAt,
			int exchangeCount) {
	}
}
