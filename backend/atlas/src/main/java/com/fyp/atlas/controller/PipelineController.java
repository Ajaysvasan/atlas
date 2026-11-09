package com.fyp.atlas.controller;

import java.util.List;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;

import com.fyp.atlas.dto.IngestResponseDto;
import com.fyp.atlas.dto.QueryRequestDto;
import com.fyp.atlas.dto.QueryResponseDto;
import com.fyp.atlas.security.AuthenticatedUser;
import com.fyp.atlas.service.IngestService;
import com.fyp.atlas.service.QueryService;
import com.fyp.atlas.web.RequestCorrelation;

import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

/** /query and /ingest: the user is always the token's subject, never a request field. */
@RestController
public class PipelineController {

	private final QueryService queries;
	private final IngestService ingestion;

	public PipelineController(QueryService queries, IngestService ingestion) {
		this.queries = queries;
		this.ingestion = ingestion;
	}

	@PostMapping(path = "/query", consumes = MediaType.APPLICATION_JSON_VALUE)
	public QueryResponseDto query(@Valid @RequestBody QueryRequestDto body, Authentication authentication,
			HttpServletRequest request) {
		return queries.ask(AuthenticatedUser.id(authentication), body, RequestCorrelation.of(request));
	}

	@PostMapping(path = "/ingest", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<IngestResponseDto> ingest(@RequestPart("files") List<MultipartFile> files,
			Authentication authentication, HttpServletRequest request) {
		IngestService.Outcome outcome = ingestion.ingest(AuthenticatedUser.id(authentication), files,
				RequestCorrelation.of(request));
		return ResponseEntity.status(outcome.status()).body(outcome.body());
	}
}
