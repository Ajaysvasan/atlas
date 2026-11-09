package com.fyp.atlas.security;

import java.util.UUID;

import com.fyp.atlas.exception.ApiException;
import com.fyp.atlas.exception.ErrorCode;

import org.springframework.security.core.Authentication;
import org.springframework.security.oauth2.server.resource.authentication.JwtAuthenticationToken;

/** The caller's user id: only ever the subject of a verified token, never a request field. */
public final class AuthenticatedUser {

	private AuthenticatedUser() {
	}

	public static UUID id(Authentication authentication) {
		if (authentication instanceof JwtAuthenticationToken token && token.isAuthenticated()) {
			try {
				return UUID.fromString(token.getToken().getSubject());
			}
			catch (IllegalArgumentException | NullPointerException error) {
				throw new ApiException(ErrorCode.INVALID_TOKEN);
			}
		}
		throw new ApiException(ErrorCode.UNAUTHORIZED);
	}
}
