package com.fyp.atlas.service;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import com.fyp.atlas.dto.IngestResponseDto.FileResultDto;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;

class UploadRulesTest {

	@ParameterizedTest
	@CsvSource(delimiter = '|', value = { "report.pdf|report.pdf", "../../etc/passwd|passwd",
			"..\\..\\windows\\system.ini|system.ini", "/abs/notes.md|notes.md", "'   '|upload", "..|upload",
			"a:b*c?.txt|a_b_c_.txt" })
	void filenamesLoseDirectoriesAndPathSyntax(String raw, String safe) {
		assertThat(UploadValidator.safeFilename(raw)).isEqualTo(safe);
	}

	@Test
	void controlCharactersAreReplacedAndLongNamesKeepTheirExtension() {
		assertThat(UploadValidator.safeFilename("bad\u0000name\n.txt")).isEqualTo("bad_name_.txt");
		String longName = UploadValidator.safeFilename("a".repeat(500) + ".docx");
		assertThat(longName).hasSizeLessThanOrEqualTo(200).endsWith(".docx");
		assertThat(UploadValidator.safeFilename(null)).isEqualTo("upload");
	}

	static FileResultDto result(String status, String code) {
		return new FileResultDto(0, "f", status, null, null, null, null, code, null);
	}

	@Test
	void theHttpStatusExplainsTheOutcome() {
		assertThat(IngestService.httpStatus("COMPLETED", List.of(result("INGESTED", null)))).isEqualTo(HttpStatus.OK);
		assertThat(IngestService.httpStatus("PARTIAL", List.of())).isEqualTo(HttpStatus.MULTI_STATUS);
		assertThat(IngestService.httpStatus("FAILED", List.of(result("NOT_INGESTED", "QUOTA_EXCEEDED"),
				result("REJECTED", "UNSUPPORTED_TYPE")))).isEqualTo(HttpStatus.INSUFFICIENT_STORAGE);
		assertThat(IngestService.httpStatus("FAILED", List.of(result("REJECTED", "TOO_LARGE"))))
			.isEqualTo(HttpStatus.CONTENT_TOO_LARGE);
		assertThat(IngestService.httpStatus("FAILED", List.of(result("REJECTED", "UNSUPPORTED_TYPE"))))
			.isEqualTo(HttpStatus.UNSUPPORTED_MEDIA_TYPE);
		assertThat(IngestService.httpStatus("FAILED", List.of(result("NOT_INGESTED", "VECTOR_STORE_UNAVAILABLE"))))
			.isEqualTo(HttpStatus.SERVICE_UNAVAILABLE);
		assertThat(IngestService.httpStatus("FAILED", List.of(result("REJECTED", "MALFORMED"),
				result("REJECTED", "UNSUPPORTED_TYPE")))).isEqualTo(HttpStatus.UNPROCESSABLE_CONTENT);
	}
}
