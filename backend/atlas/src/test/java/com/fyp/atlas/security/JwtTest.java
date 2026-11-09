package com.fyp.atlas.security;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.Base64;
import java.util.List;
import java.util.UUID;

import javax.crypto.SecretKey;

import com.fyp.atlas.config.AtlasProperties;
import com.fyp.atlas.dto.TokenResponse;

import org.junit.jupiter.api.Test;
import org.springframework.security.oauth2.jose.jws.MacAlgorithm;
import org.springframework.security.oauth2.jwt.JwsHeader;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.security.oauth2.jwt.JwtClaimsSet;
import org.springframework.security.oauth2.jwt.JwtDecoder;
import org.springframework.security.oauth2.jwt.JwtEncoder;
import org.springframework.security.oauth2.jwt.JwtEncoderParameters;
import org.springframework.security.oauth2.jwt.JwtValidationException;
import org.springframework.security.oauth2.jwt.BadJwtException;

class JwtTest {

	static final String SECRET = Base64.getEncoder().encodeToString("0123456789abcdef0123456789abcdef!".getBytes());
	static final AtlasProperties.Jwt SETTINGS = new AtlasProperties.Jwt(SECRET, "atlas", "atlas-api",
			Duration.ofMinutes(30));

	final JwtConfig config = new JwtConfig();
	final SecretKey key = JwtConfig.signingKey(SECRET);
	final JwtEncoder encoder = config.jwtEncoder(key);
	final JwtDecoder decoder = config.jwtDecoder(key, properties(SETTINGS));

	static AtlasProperties properties(AtlasProperties.Jwt jwt) {
		return new AtlasProperties(jwt, null, null, null, null, null, null);
	}

	JwtTokenService service(Clock clock) {
		return new JwtTokenService(encoder, properties(SETTINGS), clock);
	}

	@Test
	void issuedTokenCarriesSubjectIssuerAudienceAndConfiguredExpiry() {
		UUID user = UUID.randomUUID();
		TokenResponse token = service(Clock.systemUTC()).issue(user);
		assertThat(token.tokenType()).isEqualTo("Bearer");
		assertThat(token.expiresIn()).isEqualTo(1800);

		Jwt jwt = decoder.decode(token.accessToken());
		assertThat(jwt.getSubject()).isEqualTo(user.toString());
		assertThat(jwt.getClaimAsString("iss")).isEqualTo("atlas");
		assertThat(jwt.getAudience()).containsExactly("atlas-api");
		assertThat(Duration.between(jwt.getIssuedAt(), jwt.getExpiresAt())).isEqualTo(Duration.ofMinutes(30));
		assertThat(jwt.getId()).isNotBlank();
		assertThat(jwt.getHeaders().get("alg").toString()).isEqualTo("HS256");
	}

	@Test
	void expiredTokenIsRejected() {
		Clock past = Clock.fixed(Instant.now().minus(Duration.ofHours(2)), ZoneOffset.UTC);
		String token = service(past).issue(UUID.randomUUID()).accessToken();
		assertThatThrownBy(() -> decoder.decode(token)).isInstanceOf(JwtValidationException.class)
			.hasMessageContaining("expired");
	}

	@Test
	void wrongIssuerOrAudienceIsRejected() {
		assertThatThrownBy(() -> decoder.decode(sign("someone-else", "atlas-api", UUID.randomUUID().toString())))
			.isInstanceOf(JwtValidationException.class);
		assertThatThrownBy(() -> decoder.decode(sign("atlas", "another-api", UUID.randomUUID().toString())))
			.isInstanceOf(JwtValidationException.class);
	}

	@Test
	void subjectMustBeAUserUuid() {
		assertThatThrownBy(() -> decoder.decode(sign("atlas", "atlas-api", "admin")))
			.isInstanceOf(JwtValidationException.class);
	}

	@Test
	void tokenSignedWithAnotherKeyIsRejected() {
		SecretKey other = JwtConfig.signingKey(Base64.getEncoder().encodeToString(new byte[48]));
		String forged = new JwtConfig().jwtEncoder(other)
			.encode(JwtEncoderParameters.from(JwsHeader.with(MacAlgorithm.HS256).build(),
					claims("atlas", "atlas-api", UUID.randomUUID().toString())))
			.getTokenValue();
		assertThatThrownBy(() -> decoder.decode(forged)).isInstanceOf(BadJwtException.class);
	}

	@Test
	void tamperedPayloadIsRejected() {
		String token = service(Clock.systemUTC()).issue(UUID.randomUUID()).accessToken();
		String[] parts = token.split("\\.");
		String payload = new String(Base64.getUrlDecoder().decode(parts[1]));
		String swapped = payload.replaceFirst("\"sub\":\"[^\"]+\"", "\"sub\":\"" + UUID.randomUUID() + "\"");
		String tampered = parts[0] + "." + Base64.getUrlEncoder().withoutPadding().encodeToString(swapped.getBytes())
				+ "." + parts[2];
		assertThatThrownBy(() -> decoder.decode(tampered)).isInstanceOf(BadJwtException.class);
	}

	@Test
	void signingKeyMustBePresentBase64AndLongEnough() {
		assertThatThrownBy(() -> JwtConfig.signingKey(null)).hasMessageContaining("ATLAS_JWT_SECRET is not set");
		assertThatThrownBy(() -> JwtConfig.signingKey("  ")).hasMessageContaining("not set");
		assertThatThrownBy(() -> JwtConfig.signingKey("not base64 !!")).hasMessageContaining("base64");
		assertThatThrownBy(() -> JwtConfig.signingKey(Base64.getEncoder().encodeToString(new byte[16])))
			.hasMessageContaining("at least 32 bytes");
	}

	String sign(String issuer, String audience, String subject) {
		return encoder.encode(JwtEncoderParameters.from(JwsHeader.with(MacAlgorithm.HS256).build(),
				claims(issuer, audience, subject))).getTokenValue();
	}

	static JwtClaimsSet claims(String issuer, String audience, String subject) {
		Instant now = Instant.now();
		return JwtClaimsSet.builder()
			.issuer(issuer)
			.audience(List.of(audience))
			.subject(subject)
			.issuedAt(now)
			.expiresAt(now.plusSeconds(600))
			.build();
	}
}
