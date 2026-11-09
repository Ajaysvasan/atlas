package com.fyp.atlas.service;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import com.fyp.atlas.config.AtlasProperties;
import com.fyp.atlas.dto.IngestResponseDto;
import com.fyp.atlas.dto.IngestResponseDto.FileResultDto;
import com.fyp.atlas.exception.ApiException;
import com.fyp.atlas.exception.ErrorCode;
import com.fyp.atlas.grpc.AtlasGrpcClient;
import com.fyp.atlas.grpc.AtlasGrpcClient.UploadSource;
import com.fyp.atlas.grpc.v1.FileResult;
import com.fyp.atlas.grpc.v1.IngestResponse;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

/**
 * Checks each upload, streams the acceptable ones to the pipeline's ingestion (which owns the
 * quota and the corpus), and merges both verdicts back into upload order.
 */
@Service
public class IngestService {

	private static final Logger log = LoggerFactory.getLogger(IngestService.class);

	public record Outcome(HttpStatus status, IngestResponseDto body) {
	}

	private final AtlasGrpcClient pipeline;
	private final UploadValidator validator;
	private final AuthService auth;
	private final AtlasProperties.Upload settings;

	public IngestService(AtlasGrpcClient pipeline, UploadValidator validator, AuthService auth,
			AtlasProperties properties) {
		this.pipeline = pipeline;
		this.validator = validator;
		this.auth = auth;
		this.settings = properties.upload();
	}

	public Outcome ingest(UUID userId, List<MultipartFile> files, String requestId) {
		long started = System.nanoTime();
		if (files == null || files.isEmpty()) {
			throw new ApiException(ErrorCode.VALIDATION_FAILED, "No files were uploaded.");
		}
		if (files.size() > settings.maxFiles()) {
			throw new ApiException(ErrorCode.VALIDATION_FAILED,
					"At most " + settings.maxFiles() + " files may be uploaded at once.");
		}
		long total = files.stream().mapToLong(MultipartFile::getSize).sum();
		if (total > settings.maxRequestSize().toBytes()) {
			throw new ApiException(ErrorCode.PAYLOAD_TOO_LARGE);
		}
		auth.requireActive(userId);

		FileResultDto[] results = new FileResultDto[files.size()];
		List<UploadSource> accepted = new ArrayList<>();
		List<Integer> positions = new ArrayList<>();
		for (int i = 0; i < files.size(); i++) {
			MultipartFile file = files.get(i);
			String name = UploadValidator.safeFilename(file.getOriginalFilename());
			Optional<UploadValidator.Rejection> rejection = validator.validate(file, name);
			if (rejection.isPresent()) {
				results[i] = new FileResultDto(i, name, "REJECTED", null, null, null, null, rejection.get().code(),
						rejection.get().message());
				continue;
			}
			accepted.add(new UploadSource(name, file.getContentType(), file.getSize(), file::getInputStream));
			positions.add(i);
		}
		log.info("Upload {}: {} file(s), {} accepted for ingestion", requestId, files.size(), accepted.size());

		Long used = null;
		Long limit = null;
		if (!accepted.isEmpty()) {
			IngestResponse response = pipeline.ingest(requestId, userId, accepted);
			if (response.getFilesCount() != accepted.size()) {
				log.error("Pipeline reported {} files for {} sent", response.getFilesCount(), accepted.size());
				throw new ApiException(ErrorCode.INTERNAL_ERROR, "The Atlas pipeline returned an incomplete response.");
			}
			for (FileResult file : response.getFilesList()) {
				int position = positions.get(file.getFileIndex());
				results[position] = fromPipeline(position, file);
			}
			used = response.getStorageUsedBytes();
			limit = response.hasStorageLimitBytes() ? response.getStorageLimitBytes() : null;
		}

		List<FileResultDto> merged = List.of(results);
		int succeeded = (int) merged.stream().filter(r -> r.status().equals("INGESTED")).count();
		int failed = merged.size() - succeeded;
		String status = failed == 0 ? "COMPLETED" : succeeded == 0 ? "FAILED" : "PARTIAL";
		IngestResponseDto body = new IngestResponseDto(requestId, status, succeeded, failed, used, limit,
				(System.nanoTime() - started) / 1_000_000, merged);
		return new Outcome(httpStatus(status, merged), body);
	}

	private static FileResultDto fromPipeline(int position, FileResult file) {
		String status = switch (file.getStatus()) {
			case FILE_STATUS_INGESTED -> "INGESTED";
			case FILE_STATUS_REJECTED -> "REJECTED";
			case FILE_STATUS_NOT_INGESTED -> "NOT_INGESTED";
			default -> "FAILED";
		};
		boolean ingested = status.equals("INGESTED");
		return new FileResultDto(position, file.getFilename(), status,
				ingested ? file.getDocumentId() : null, ingested ? file.getChunks() : null,
				ingested ? file.getVectors() : null, ingested ? file.getStoredBytes() : null,
				ingested ? null : file.getErrorCode(), ingested ? null : file.getErrorMessage());
	}

	/** 200 all in, 207 some in; when none: 507 quota, 413 size, 415 type, otherwise 422. */
	static HttpStatus httpStatus(String status, List<FileResultDto> files) {
		if (status.equals("COMPLETED")) {
			return HttpStatus.OK;
		}
		if (status.equals("PARTIAL")) {
			return HttpStatus.MULTI_STATUS;
		}
		if (files.stream().anyMatch(f -> "QUOTA_EXCEEDED".equals(f.errorCode()) || "DISK_FULL".equals(f.errorCode()))) {
			return HttpStatus.INSUFFICIENT_STORAGE;
		}
		if (files.stream().allMatch(f -> "TOO_LARGE".equals(f.errorCode()))) {
			return HttpStatus.CONTENT_TOO_LARGE;
		}
		if (files.stream().allMatch(f -> "UNSUPPORTED_TYPE".equals(f.errorCode()))) {
			return HttpStatus.UNSUPPORTED_MEDIA_TYPE;
		}
		if (files.stream().anyMatch(f -> "VECTOR_STORE_UNAVAILABLE".equals(f.errorCode()))) {
			return HttpStatus.SERVICE_UNAVAILABLE;
		}
		return HttpStatus.UNPROCESSABLE_CONTENT;
	}
}
