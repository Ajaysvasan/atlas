package com.fyp.atlas.security;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Duration;

import com.fyp.atlas.config.AtlasProperties;
import com.fyp.atlas.exception.ApiException;
import com.fyp.atlas.exception.ErrorCode;

import org.junit.jupiter.api.Test;

class RateLimiterTest {

	static RateLimiter limiter(int perIp, int perEmail) {
		AtlasProperties.RateLimit limits = new AtlasProperties.RateLimit(
				new AtlasProperties.Window(perIp, Duration.ofMinutes(1)),
				new AtlasProperties.Window(perEmail, Duration.ofMinutes(15)),
				new AtlasProperties.Window(2, Duration.ofHours(1)));
		return new RateLimiter(new AtlasProperties(null, null, limits, null, null, null, null));
	}

	@Test
	void emailIsLimitedAcrossAddresses() {
		RateLimiter limiter = limiter(100, 3);
		for (int i = 0; i < 3; i++) {
			limiter.loginAttempt("10.0.0." + i, "victim@example.com");
		}
		assertThatThrownBy(() -> limiter.loginAttempt("10.0.0.99", "victim@example.com"))
			.isInstanceOfSatisfying(ApiException.class, error -> {
				assertThat(error.code()).isEqualTo(ErrorCode.TOO_MANY_REQUESTS);
				assertThat(error.retryAfter()).isPositive();
			});
		limiter.loginAttempt("10.0.0.99", "someone-else@example.com");
	}

	@Test
	void addressIsLimitedAcrossEmails() {
		RateLimiter limiter = limiter(2, 100);
		limiter.loginAttempt("10.1.1.1", "a@example.com");
		limiter.loginAttempt("10.1.1.1", "b@example.com");
		assertThatThrownBy(() -> limiter.loginAttempt("10.1.1.1", "c@example.com")).isInstanceOf(ApiException.class);
	}

	@Test
	void registrationIsLimitedPerAddress() {
		RateLimiter limiter = limiter(10, 10);
		limiter.registration("10.2.2.2");
		limiter.registration("10.2.2.2");
		assertThatThrownBy(() -> limiter.registration("10.2.2.2")).isInstanceOf(ApiException.class);
		limiter.registration("10.2.2.3");
	}
}
