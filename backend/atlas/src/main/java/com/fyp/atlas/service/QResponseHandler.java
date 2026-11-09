package com.fyp.atlas.service;

import java.util.List;

import com.fyp.atlas.dto.QueryResponseDto;
import com.fyp.atlas.exception.ApiException;
import com.fyp.atlas.exception.ErrorCode;
import com.fyp.atlas.grpc.v1.QueryResponse;
import com.fyp.atlas.grpc.v1.QueryStatus;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

/**
 * The internal qResponse boundary: Python's gRPC QueryResponse in, the public REST response out. It is a
 * service, not an HTTP route, so nothing outside this process can inject an answer.
 */
@Service
public class QResponseHandler {

	private static final Logger log = LoggerFactory.getLogger(QResponseHandler.class);

	public QueryResponseDto handle(String requestId, QueryResponse response) {
		if (!requestId.equals(response.getRequestId())) {
			log.error("Pipeline answered request {} with id {}", requestId, response.getRequestId());
			throw new ApiException(ErrorCode.INTERNAL_ERROR, "The Atlas pipeline returned a mismatched response.");
		}
		if (response.getConversationId().isBlank() || response.getAnswer().isBlank()) {
			log.error("Pipeline response for {} lacks a conversation or an answer", requestId);
			throw new ApiException(ErrorCode.INTERNAL_ERROR, "The Atlas pipeline returned an incomplete response.");
		}
		String status = switch (response.getStatus()) {
			case QUERY_STATUS_COMPLETED -> "COMPLETED";
			case QUERY_STATUS_COMPLETED_WITH_WARNINGS -> "COMPLETED_WITH_WARNINGS";
			default -> {
				log.error("Pipeline response for {} has status {}", requestId, response.getStatus());
				throw new ApiException(ErrorCode.INTERNAL_ERROR, "The Atlas pipeline returned an unknown status.");
			}
		};
		log.info("Answered request {} in conversation {} ({} ms, {} retries, {})", requestId,
				response.getConversationId(), response.getExecutionTimeMs(), response.getRetryCount(), status);
		return new QueryResponseDto(response.getRequestId(), response.getConversationId(),
				response.getConversationName(), response.getNewConversation(), response.getAnswer(), status,
				response.getExecutionTimeMs(), response.getRetryCount(), List.copyOf(response.getWarningsList()),
				response.getPersisted());
	}

	static boolean known(QueryStatus status) {
		return status == QueryStatus.QUERY_STATUS_COMPLETED
				|| status == QueryStatus.QUERY_STATUS_COMPLETED_WITH_WARNINGS;
	}
}
