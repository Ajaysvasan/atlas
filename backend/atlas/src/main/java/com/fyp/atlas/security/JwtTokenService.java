package com.fyp.atlas.security;

import java.time.Clock;
import java.time.Instant;
import java.util.List;
import java.util.UUID;

import com.fyp.atlas.config.AtlasProperties;
import com.fyp.atlas.dto.TokenResponse;

import org.springframework.security.oauth2.jose.jws.MacAlgorithm;
import org.springframework.security.oauth2.jwt.JwsHeader;
import org.springframework.security.oauth2.jwt.JwtClaimsSet;
import org.springframework.security.oauth2.jwt.JwtEncoder;
import org.springframework.security.oauth2.jwt.JwtEncoderParameters;
import org.springframework.stereotype.Service;

/** Issues access tokens: sub = user id, plus iss, aud, iat, exp and a unique jti. */
@Service
public class JwtTokenService {

	private final JwtEncoder encoder;
	private final AtlasProperties.Jwt settings;
	private final Clock clock;

	public JwtTokenService(JwtEncoder encoder, AtlasProperties properties, Clock clock) {
		this.encoder = encoder;
		this.settings = properties.jwt();
		this.clock = clock;
	}

	public TokenResponse issue(UUID userId) {
		Instant now = clock.instant();
		Instant expires = now.plus(settings.expiration());
		JwtClaimsSet claims = JwtClaimsSet.builder()
			.subject(userId.toString())
			.issuer(settings.issuer())
			.audience(List.of(settings.audience()))
			.issuedAt(now)
			.notBefore(now)
			.expiresAt(expires)
			.id(UUID.randomUUID().toString())
			.build();
		JwsHeader header = JwsHeader.with(MacAlgorithm.HS256).type("JWT").build();
		String token = encoder.encode(JwtEncoderParameters.from(header, claims)).getTokenValue();
		return TokenResponse.bearer(token, settings.expiration().toSeconds());
	}
}
