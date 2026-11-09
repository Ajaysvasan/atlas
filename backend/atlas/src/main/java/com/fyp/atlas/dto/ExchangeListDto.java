package com.fyp.atlas.dto;

import java.time.OffsetDateTime;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonInclude;

/** GET /conversations/{id}/exchanges: one page of a conversation's exchanges, oldest first. */
public record ExchangeListDto(
		String requestId,
		String conversationId,
		String conversationName,
		List<Item> exchanges,
		String nextCursor) {

	/** answer and answeredAt are absent for a query with no recorded answer. */
	@JsonInclude(JsonInclude.Include.NON_NULL)
	public record Item(String exchangeId, String query, OffsetDateTime askedAt, String answer,
			OffsetDateTime answeredAt) {
	}
}
