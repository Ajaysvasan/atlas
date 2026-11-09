package com.fyp.atlas.security;

import java.time.Duration;
import java.util.concurrent.TimeUnit;

import com.fyp.atlas.config.AtlasProperties;
import com.fyp.atlas.exception.ApiException;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import io.github.bucket4j.Bandwidth;
import io.github.bucket4j.Bucket;
import io.github.bucket4j.ConsumptionProbe;

import org.springframework.stereotype.Component;

/**
 * Token buckets per client address and per login email, in this process's memory. Entries expire
 * once idle, so the map cannot be grown without bound by spraying keys.
 */
@Component
public class RateLimiter {

	private static final long MAX_KEYS = 100_000;

	private final AtlasProperties.RateLimit settings;
	private final Cache<String, Bucket> buckets = Caffeine.newBuilder()
		.maximumSize(MAX_KEYS)
		.expireAfterAccess(2, TimeUnit.HOURS)
		.build();

	public RateLimiter(AtlasProperties properties) {
		this.settings = properties.rateLimit();
	}

	public void loginAttempt(String clientAddress, String normalisedEmail) {
		consume("login-ip:" + clientAddress, settings.loginPerIp());
		consume("login-email:" + normalisedEmail, settings.loginPerEmail());
	}

	public void registration(String clientAddress) {
		consume("register-ip:" + clientAddress, settings.registerPerIp());
	}

	private void consume(String key, AtlasProperties.Window window) {
		Bucket bucket = buckets.get(key, k -> Bucket.builder()
			.addLimit(Bandwidth.builder()
				.capacity(window.capacity())
				.refillGreedy(window.capacity(), window.period())
				.build())
			.build());
		ConsumptionProbe probe = bucket.tryConsumeAndReturnRemaining(1);
		if (!probe.isConsumed()) {
			throw ApiException.tooManyRequests(Duration.ofNanos(probe.getNanosToWaitForRefill()));
		}
	}
}
