package com.fyp.atlas.service;

import java.security.SecureRandom;
import java.util.Base64;
import java.util.Locale;
import java.util.Optional;
import java.util.UUID;

import com.fyp.atlas.dto.LoginRequest;
import com.fyp.atlas.dto.RegisterRequest;
import com.fyp.atlas.dto.RegisterResponse;
import com.fyp.atlas.dto.TokenResponse;
import com.fyp.atlas.entity.UserAccount;
import com.fyp.atlas.exception.ApiException;
import com.fyp.atlas.exception.ErrorCode;
import com.fyp.atlas.repository.UserRepository;
import com.fyp.atlas.security.JwtTokenService;
import com.fyp.atlas.security.RateLimiter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

/** Registration and login. Passwords and hashes are never logged or returned. */
@Service
public class AuthService {

	private static final Logger log = LoggerFactory.getLogger(AuthService.class);

	private final UserRepository users;
	private final PasswordEncoder passwords;
	private final JwtTokenService tokens;
	private final RateLimiter rateLimiter;
	/** Verified against when the email is unknown, so both outcomes cost one Argon2 run. */
	private final String decoyHash;

	public AuthService(UserRepository users, PasswordEncoder passwords, JwtTokenService tokens,
			RateLimiter rateLimiter) {
		this.users = users;
		this.passwords = passwords;
		this.tokens = tokens;
		this.rateLimiter = rateLimiter;
		byte[] random = new byte[24];
		new SecureRandom().nextBytes(random);
		this.decoyHash = passwords.encode(Base64.getEncoder().encodeToString(random));
	}

	public static String normaliseEmail(String email) {
		return email == null ? "" : email.trim().toLowerCase(Locale.ROOT);
	}

	public RegisterResponse register(RegisterRequest request, String clientAddress) {
		rateLimiter.registration(clientAddress);
		String username = request.username().trim();
		String email = normaliseEmail(request.email());
		if (users.existsByUsername(username)) {
			throw new ApiException(ErrorCode.USERNAME_TAKEN);
		}
		if (users.existsByEmail(email)) {
			throw new ApiException(ErrorCode.EMAIL_TAKEN);
		}
		String hash = passwords.encode(request.password());
		UserAccount account;
		try {
			account = users.insert(UUID.randomUUID(), username, email, hash);
		}
		catch (DuplicateKeyException race) {
			String message = String.valueOf(race.getMostSpecificCause().getMessage());
			throw new ApiException(message.contains(UserRepository.USERNAME_KEY) ? ErrorCode.USERNAME_TAKEN
					: ErrorCode.EMAIL_TAKEN);
		}
		log.info("Registered user {}", account.userId());
		return new RegisterResponse(account.userId(), account.username(), account.email(), account.createdAt());
	}

	public TokenResponse login(LoginRequest request, String clientAddress) {
		String email = normaliseEmail(request.email());
		rateLimiter.loginAttempt(clientAddress, email);
		Optional<UserAccount> found = users.findByEmail(email);
		String hash = found.map(UserAccount::passwordHash).orElse(decoyHash);
		boolean matches = passwords.matches(request.password(), hash);
		if (found.isEmpty() || !matches || !found.get().active()) {
			log.info("Login refused from {}", clientAddress);
			throw new ApiException(ErrorCode.INVALID_CREDENTIALS);
		}
		UserAccount account = found.get();
		if (passwords.upgradeEncoding(account.passwordHash())) {
			boolean replaced = users.replacePasswordHash(account.userId(), account.passwordHash(),
					passwords.encode(request.password()));
			log.info("Rehashed the password of user {} with the current Argon2id parameters: {}", account.userId(),
					replaced);
		}
		log.info("User {} logged in", account.userId());
		return tokens.issue(account.userId());
	}

	/** A token outlives a deactivation by at most its expiry; the long operations check again here. */
	public void requireActive(UUID userId) {
		if (!users.isActive(userId)) {
			throw new ApiException(ErrorCode.INVALID_TOKEN);
		}
	}
}
