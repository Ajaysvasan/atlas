package com.fyp.atlas.service;

import java.util.UUID;

import com.fyp.atlas.config.AtlasProperties;
import com.fyp.atlas.dto.QueryRequestDto;
import com.fyp.atlas.dto.QueryResponseDto;
import com.fyp.atlas.exception.ApiException;
import com.fyp.atlas.exception.ErrorCode;
import com.fyp.atlas.grpc.AtlasGrpcClient;
import com.fyp.atlas.grpc.v1.QueryRequest;
import com.fyp.atlas.grpc.v1.QueryResponse;

import org.springframework.stereotype.Service;

/**
 * A query for the authenticated user. Ownership of an existing conversation is checked by the
 * pipeline against its conversation registry, the one authoritative record; another user's id
 * comes back as not found.
 */
@Service
public class QueryService {

	private final AtlasGrpcClient pipeline;
	private final QResponseHandler responses;
	private final AuthService auth;
	private final int maxChars;

	public QueryService(AtlasGrpcClient pipeline, QResponseHandler responses, AuthService auth,
			AtlasProperties properties) {
		this.pipeline = pipeline;
		this.responses = responses;
		this.auth = auth;
		this.maxChars = properties.query().maxChars();
	}

	public QueryResponseDto ask(UUID userId, QueryRequestDto body, String requestId) {
		String query = body.query().strip();
		if (query.isEmpty() || query.length() > maxChars) {
			throw new ApiException(ErrorCode.VALIDATION_FAILED, "The query must be 1-" + maxChars + " characters.");
		}
		auth.requireActive(userId);
		QueryRequest.Builder request = QueryRequest.newBuilder()
			.setRequestId(requestId)
			.setUserId(userId.toString())
			.setQuery(query)
			.putMetadata("client", "rest");
		if (body.conversationId() != null && !body.conversationId().isBlank()) {
			request.setConversationId(body.conversationId());
		}
		QueryResponse response = pipeline.query(request.build());
		return responses.handle(requestId, response);
	}
}
