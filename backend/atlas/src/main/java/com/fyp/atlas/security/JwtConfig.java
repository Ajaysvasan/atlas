package com.fyp.atlas.security;

import java.util.Base64;
import java.util.UUID;

import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;

import com.fyp.atlas.config.AtlasProperties;
import com.nimbusds.jose.jwk.source.ImmutableSecret;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.oauth2.core.DelegatingOAuth2TokenValidator;
import org.springframework.security.oauth2.core.OAuth2Error;
import org.springframework.security.oauth2.core.OAuth2TokenValidator;
import org.springframework.security.oauth2.core.OAuth2TokenValidatorResult;
import org.springframework.security.oauth2.jose.jws.MacAlgorithm;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.security.oauth2.jwt.JwtClaimNames;
import org.springframework.security.oauth2.jwt.JwtClaimValidator;
import org.springframework.security.oauth2.jwt.JwtDecoder;
import org.springframework.security.oauth2.jwt.JwtEncoder;
import org.springframework.security.oauth2.jwt.JwtValidators;
import org.springframework.security.oauth2.jwt.NimbusJwtDecoder;
import org.springframework.security.oauth2.jwt.NimbusJwtEncoder;

/** HS256 tokens signed with a key from the environment; issuer, audience, expiry and a UUID subject are required. */
@Configuration(proxyBeanMethods = false)
public class JwtConfig {

	static final int MIN_KEY_BYTES = 32;

	@Bean
	SecretKey jwtSigningKey(AtlasProperties properties) {
		return signingKey(properties.jwt().secret());
	}

	@Bean
	JwtEncoder jwtEncoder(SecretKey jwtSigningKey) {
		return new NimbusJwtEncoder(new ImmutableSecret<>(jwtSigningKey));
	}

	@Bean
	JwtDecoder jwtDecoder(SecretKey jwtSigningKey, AtlasProperties properties) {
		NimbusJwtDecoder decoder = NimbusJwtDecoder.withSecretKey(jwtSigningKey)
			.macAlgorithm(MacAlgorithm.HS256)
			.build();
		decoder.setJwtValidator(validator(properties.jwt()));
		return decoder;
	}

	static OAuth2TokenValidator<Jwt> validator(AtlasProperties.Jwt settings) {
		OAuth2TokenValidator<Jwt> audience = jwt -> jwt.getAudience() != null
				&& jwt.getAudience().contains(settings.audience())
						? OAuth2TokenValidatorResult.success()
						: OAuth2TokenValidatorResult
							.failure(new OAuth2Error("invalid_token", "The token audience is not accepted", null));
		OAuth2TokenValidator<Jwt> expiry = new JwtClaimValidator<>(JwtClaimNames.EXP, exp -> exp != null);
		OAuth2TokenValidator<Jwt> subject = new JwtClaimValidator<String>(JwtClaimNames.SUB, JwtConfig::isUuid);
		return new DelegatingOAuth2TokenValidator<>(JwtValidators.createDefaultWithIssuer(settings.issuer()),
				audience, expiry, subject);
	}

	static boolean isUuid(String value) {
		if (value == null) {
			return false;
		}
		try {
			return UUID.fromString(value).toString().equals(value);
		}
		catch (IllegalArgumentException error) {
			return false;
		}
	}

	/** Refuses to start without a real secret, rather than signing with something guessable. */
	static SecretKey signingKey(String base64) {
		if (base64 == null || base64.isBlank()) {
			throw new IllegalStateException(
					"ATLAS_JWT_SECRET is not set: give it base64 of at least 32 random bytes (openssl rand -base64 48)");
		}
		byte[] key;
		try {
			key = Base64.getDecoder().decode(base64.trim());
		}
		catch (IllegalArgumentException error) {
			throw new IllegalStateException("ATLAS_JWT_SECRET is not valid base64", error);
		}
		if (key.length < MIN_KEY_BYTES) {
			throw new IllegalStateException("ATLAS_JWT_SECRET must decode to at least " + MIN_KEY_BYTES + " bytes");
		}
		return new SecretKeySpec(key, "HmacSHA256");
	}
}
