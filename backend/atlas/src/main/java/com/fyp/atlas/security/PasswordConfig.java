package com.fyp.atlas.security;

import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

import com.fyp.atlas.config.AtlasProperties;
import com.fyp.atlas.exception.ApiException;
import com.fyp.atlas.exception.ErrorCode;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.crypto.argon2.Argon2PasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

/**
 * Argon2id with a fresh random salt per password (BouncyCastle). The encoded hash carries its own
 * parameters and salt, so raising the work factor later only needs {@code upgradeEncoding} on login.
 */
@Configuration(proxyBeanMethods = false)
public class PasswordConfig {

	@Bean
	public PasswordEncoder passwordEncoder(AtlasProperties properties) {
		return new Bounded(argon2(properties.argon2()), properties.argon2().maxConcurrent());
	}

	public static Argon2PasswordEncoder argon2(AtlasProperties.Argon2 settings) {
		return new Argon2PasswordEncoder(settings.saltLength(), settings.hashLength(), settings.parallelism(),
				settings.memoryKib(), settings.iterations());
	}

	/** Each hash holds memoryKib of RAM while it runs; this caps how many run at once. */
	static final class Bounded implements PasswordEncoder {

		private static final long WAIT_SECONDS = 10;

		private final PasswordEncoder delegate;
		private final Semaphore permits;

		Bounded(PasswordEncoder delegate, int maxConcurrent) {
			this.delegate = delegate;
			this.permits = new Semaphore(maxConcurrent, true);
		}

		private <T> T bounded(Supplier<T> work) {
			try {
				if (!permits.tryAcquire(WAIT_SECONDS, TimeUnit.SECONDS)) {
					throw new ApiException(ErrorCode.BACKEND_BUSY);
				}
			}
			catch (InterruptedException error) {
				Thread.currentThread().interrupt();
				throw new ApiException(ErrorCode.BACKEND_BUSY);
			}
			try {
				return work.get();
			}
			finally {
				permits.release();
			}
		}

		@Override
		public String encode(CharSequence rawPassword) {
			return bounded(() -> delegate.encode(rawPassword));
		}

		@Override
		public boolean matches(CharSequence rawPassword, String encodedPassword) {
			return bounded(() -> delegate.matches(rawPassword, encodedPassword));
		}

		@Override
		public boolean upgradeEncoding(String encodedPassword) {
			return delegate.upgradeEncoding(encodedPassword);
		}
	}
}
