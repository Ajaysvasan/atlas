package com.fyp.atlas.dto;

/** A signed access token; expiresIn is in seconds, from the configured expiration. */
public record TokenResponse(String accessToken, String tokenType, long expiresIn) {

	public static TokenResponse bearer(String token, long expiresIn) {
		return new TokenResponse(token, "Bearer", expiresIn);
	}

	@Override
	public String toString() {
		return "TokenResponse[tokenType=" + tokenType + ", expiresIn=" + expiresIn + "]";
	}
}
