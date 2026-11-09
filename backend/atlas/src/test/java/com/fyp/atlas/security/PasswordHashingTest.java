package com.fyp.atlas.security;

import static org.assertj.core.api.Assertions.assertThat;

import com.fyp.atlas.config.AtlasProperties;

import org.junit.jupiter.api.Test;
import org.springframework.security.crypto.argon2.Argon2PasswordEncoder;

class PasswordHashingTest {

	static final AtlasProperties.Argon2 CONFIGURED = new AtlasProperties.Argon2(16, 32, 1, 65536, 3, 4);

	final Argon2PasswordEncoder encoder = PasswordConfig.argon2(CONFIGURED);

	@Test
	void encodesArgon2idWithItsParametersAndSalt() {
		String hash = encoder.encode("correct horse battery staple");
		assertThat(hash).startsWith("$argon2id$v=19$m=65536,t=3,p=1$");
		assertThat(hash.split("\\$")).hasSize(6);
	}

	@Test
	void everyPasswordGetsItsOwnSalt() {
		String first = encoder.encode("same password");
		String second = encoder.encode("same password");
		assertThat(first).isNotEqualTo(second);
		assertThat(first.split("\\$")[4]).isNotEqualTo(second.split("\\$")[4]);
	}

	@Test
	void verifiesTheRightPasswordOnly() {
		String hash = encoder.encode("s3cret-Passw0rd");
		assertThat(encoder.matches("s3cret-Passw0rd", hash)).isTrue();
		assertThat(encoder.matches("s3cret-passw0rd", hash)).isFalse();
		assertThat(encoder.matches("", hash)).isFalse();
	}

	@Test
	void weakerOrOlderParametersAskForARehash() {
		Argon2PasswordEncoder older = PasswordConfig.argon2(new AtlasProperties.Argon2(16, 32, 1, 19456, 2, 4));
		String old = older.encode("password1234");
		assertThat(encoder.upgradeEncoding(old)).isTrue();
		assertThat(encoder.upgradeEncoding(encoder.encode("password1234"))).isFalse();
		assertThat(encoder.matches("password1234", old)).isTrue();
	}

	/** Measured, not assumed: the configured work factor on this machine (printed for the report). */
	@Test
	void configuredWorkFactorCostsMillisecondsNotSeconds() {
		encoder.encode("warm-up-password");
		int runs = 10;
		long started = System.nanoTime();
		String hash = null;
		for (int i = 0; i < runs; i++) {
			hash = encoder.encode("benchmark-password-" + i);
		}
		long encodeMs = (System.nanoTime() - started) / 1_000_000 / runs;
		started = System.nanoTime();
		for (int i = 0; i < runs; i++) {
			encoder.matches("benchmark-password-9", hash);
		}
		long verifyMs = (System.nanoTime() - started) / 1_000_000 / runs;
		System.out.printf("Argon2id m=65536KiB t=3 p=1: encode %d ms, verify %d ms%n", encodeMs, verifyMs);
		assertThat(encodeMs).isBetween(1L, 2000L);
		assertThat(verifyMs).isBetween(1L, 2000L);
	}
}
