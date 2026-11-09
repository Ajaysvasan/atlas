package com.fyp.atlas.dto;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * The result of POST /ingest, file by file, in upload order. Returned whatever the outcome — complete,
 * partial or failed — so the client always learns what happened to each file.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record IngestResponseDto(
		String requestId,
		String status,
		int succeeded,
		int failed,
		Long storageUsedBytes,
		Long storageLimitBytes,
		long executionTimeMs,
		List<FileResultDto> files) {

	@JsonInclude(JsonInclude.Include.NON_NULL)
	public record FileResultDto(
			int index,
			String filename,
			String status,
			String documentId,
			Integer chunks,
			Integer vectors,
			Long storedBytes,
			String errorCode,
			String message) {
	}
}
