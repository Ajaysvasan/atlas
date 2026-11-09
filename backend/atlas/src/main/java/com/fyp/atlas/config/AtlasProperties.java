package com.fyp.atlas.config;

import java.time.Duration;
import java.util.List;
import java.util.Set;

import jakarta.validation.Valid;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Positive;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;
import org.springframework.util.unit.DataSize;
import org.springframework.validation.annotation.Validated;

/** Every Atlas setting, bound from {@code atlas.*}; secrets come from the environment only. */
@Validated
@ConfigurationProperties("atlas")
public record AtlasProperties(
		@Valid @NotNull Jwt jwt,
		@Valid @NotNull Argon2 argon2,
		@Valid @NotNull RateLimit rateLimit,
		@Valid @NotNull Cors cors,
		@Valid @NotNull Grpc grpc,
		@Valid @NotNull Upload upload,
		@Valid @NotNull Query query) {

	/** The signing secret is base64 and at least 256 bits; it is checked at startup. */
	public record Jwt(
			String secret,
			@NotBlank @DefaultValue("atlas") String issuer,
			@NotBlank @DefaultValue("atlas-api") String audience,
			@NotNull @DefaultValue("PT1H") Duration expiration) {
	}

	/** Argon2id work factor (memory in KiB) and how many hashes may run at once, which bounds their memory. */
	public record Argon2(
			@Min(8) @DefaultValue("16") int saltLength,
			@Min(16) @DefaultValue("32") int hashLength,
			@Positive @DefaultValue("1") int parallelism,
			@Min(8192) @DefaultValue("65536") int memoryKib,
			@Positive @DefaultValue("3") int iterations,
			@Positive @DefaultValue("4") int maxConcurrent) {
	}

	public record RateLimit(
			@Valid @NotNull Window loginPerIp,
			@Valid @NotNull Window loginPerEmail,
			@Valid @NotNull Window registerPerIp) {
	}

	public record Window(@Positive int capacity, @NotNull Duration period) {
	}

	public record Cors(@NotNull @DefaultValue List<String> allowedOrigins) {
	}

	public record Grpc(
			@NotBlank @DefaultValue("127.0.0.1") String host,
			@Positive @DefaultValue("50051") int port,
			String serviceToken,
			@NotNull @DefaultValue("PT20M") Duration queryDeadline,
			@NotNull @DefaultValue("PT30M") Duration ingestDeadline,
			@NotNull @DefaultValue("PT10S") Duration healthDeadline,
			@NotNull @DefaultValue("4MB") DataSize maxInboundMessageSize,
			@NotNull @DefaultValue("256KB") DataSize chunkSize,
			@Valid @NotNull Tls tls) {
	}

	/** Off for loopback; for anything else, TLS with an optional client certificate (mTLS). */
	public record Tls(
			@DefaultValue("false") boolean enabled,
			String trustCertCollection,
			String clientCertChain,
			String clientPrivateKey,
			String authorityOverride) {
	}

	public record Upload(
			@Positive @DefaultValue("10") int maxFiles,
			@NotNull @DefaultValue("25MB") DataSize maxFileSize,
			@NotNull @DefaultValue("100MB") DataSize maxRequestSize,
			@NotNull @DefaultValue({"pdf", "docx", "pptx", "xlsx", "txt", "md", "html", "htm", "json", "csv"})
			Set<String> allowedExtensions,
			@NotNull @DefaultValue("200MB") DataSize maxUncompressedArchiveSize,
			@Positive @DefaultValue("10000") int maxArchiveEntries,
			@Positive @DefaultValue("100") int maxCompressionRatio) {
	}

	public record Query(@Positive @DefaultValue("8000") int maxChars) {
	}
}
