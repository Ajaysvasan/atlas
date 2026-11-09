package com.fyp.atlas.service;

import java.time.OffsetDateTime;
import java.time.format.DateTimeParseException;
import java.util.Map;
import java.util.UUID;
import java.util.regex.Pattern;

import com.fyp.atlas.dto.ConversationListDto;
import com.fyp.atlas.dto.ExchangeListDto;
import com.fyp.atlas.exception.ApiException;
import com.fyp.atlas.exception.ErrorCode;
import com.fyp.atlas.grpc.AtlasGrpcClient;
import com.fyp.atlas.grpc.v1.ExchangeRecord;
import com.fyp.atlas.grpc.v1.ListConversationsRequest;
import com.fyp.atlas.grpc.v1.ListConversationsResponse;
import com.fyp.atlas.grpc.v1.ListExchangesRequest;
import com.fyp.atlas.grpc.v1.ListExchangesResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

/**
 * The caller's conversations and their exchanges, read from the pipeline's registry and exchange log,
 * the one authoritative record. Another user's conversation is not found, as if absent.
 */
@Service
public class ConversationService {

	private static final Logger log = LoggerFactory.getLogger(ConversationService.class);

	public static final int DEFAULT_LIMIT = 50;
	public static final int MAX_LIMIT = 200;
	static final int MAX_CURSOR_CHARS = 512;
	/** The id grammar /query accepts; anything else cannot name a conversation. */
	static final Pattern CONVERSATION_ID = Pattern.compile("^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$");
	/** Cursors are unpadded base64url; Python decodes and checks them. */
	static final Pattern CURSOR = Pattern.compile("^[A-Za-z0-9_-]+$");

	private final AtlasGrpcClient pipeline;
	private final AuthService auth;

	public ConversationService(AtlasGrpcClient pipeline, AuthService auth) {
		this.pipeline = pipeline;
		this.auth = auth;
	}

	public ConversationListDto list(UUID userId, String limit, String cursor, String requestId) {
		int size = limit(limit);
		String position = cursor(cursor);
		auth.requireActive(userId);
		ListConversationsResponse response = pipeline.listConversations(ListConversationsRequest.newBuilder()
			.setRequestId(requestId)
			.setUserId(userId.toString())
			.setLimit(size)
			.setCursor(position)
			.build());
		matches(requestId, response.getRequestId());
		var items = response.getConversationsList().stream()
			.map(c -> new ConversationListDto.Item(c.getConversationId(), c.getConversationName(),
					time(requestId, c.getCreatedAt()), time(requestId, c.getLastActivityAt()), c.getExchangeCount()))
			.toList();
		return new ConversationListDto(requestId, items, blankToNull(response.getNextCursor()));
	}

	public ExchangeListDto exchanges(UUID userId, String conversationId, String limit, String cursor,
			String requestId) {
		if (conversationId == null || !CONVERSATION_ID.matcher(conversationId).matches()) {
			throw new ApiException(ErrorCode.CONVERSATION_NOT_FOUND);
		}
		int size = limit(limit);
		String position = cursor(cursor);
		auth.requireActive(userId);
		ListExchangesResponse response = pipeline.listExchanges(ListExchangesRequest.newBuilder()
			.setRequestId(requestId)
			.setUserId(userId.toString())
			.setConversationId(conversationId)
			.setLimit(size)
			.setCursor(position)
			.build());
		matches(requestId, response.getRequestId());
		var items = response.getExchangesList().stream().map(e -> exchange(requestId, e)).toList();
		return new ExchangeListDto(requestId, response.getConversationId(), response.getConversationName(), items,
				blankToNull(response.getNextCursor()));
	}

	private ExchangeListDto.Item exchange(String requestId, ExchangeRecord e) {
		return new ExchangeListDto.Item(e.getExchangeId(), e.getQuery(), time(requestId, e.getAskedAt()),
				e.hasAnswer() ? e.getAnswer() : null, e.hasAnsweredAt() ? time(requestId, e.getAnsweredAt()) : null);
	}

	static int limit(String raw) {
		if (raw == null || raw.isBlank()) {
			return DEFAULT_LIMIT;
		}
		try {
			int value = Integer.parseInt(raw.strip());
			if (value >= 1 && value <= MAX_LIMIT) {
				return value;
			}
		}
		catch (NumberFormatException ignored) {
			// reported below, like an out-of-range number
		}
		throw ApiException.validation(Map.of("limit", "must be a whole number from 1 to " + MAX_LIMIT));
	}

	static String cursor(String raw) {
		if (raw == null || raw.isEmpty()) {
			return "";
		}
		if (raw.length() > MAX_CURSOR_CHARS || !CURSOR.matcher(raw).matches()) {
			throw ApiException.validation(Map.of("cursor", "is not a cursor this API issued"));
		}
		return raw;
	}

	private static void matches(String requestId, String answered) {
		if (!requestId.equals(answered)) {
			log.error("Pipeline answered request {} with id {}", requestId, answered);
			throw new ApiException(ErrorCode.INTERNAL_ERROR, "The Atlas pipeline returned a mismatched response.");
		}
	}

	private static OffsetDateTime time(String requestId, String value) {
		try {
			return OffsetDateTime.parse(value);
		}
		catch (DateTimeParseException failure) {
			log.error("Pipeline response for {} has an unreadable timestamp", requestId);
			throw new ApiException(ErrorCode.INTERNAL_ERROR, "The Atlas pipeline returned an invalid listing.");
		}
	}

	private static String blankToNull(String value) {
		return value == null || value.isEmpty() ? null : value;
	}
}
