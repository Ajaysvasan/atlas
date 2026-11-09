package com.fyp.atlas.service;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Set;
import java.util.stream.Collectors;

import jakarta.validation.ConstraintViolation;
import jakarta.validation.Validation;
import jakarta.validation.Validator;

import com.fyp.atlas.dto.LoginRequest;
import com.fyp.atlas.dto.RegisterRequest;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class RequestValidationTest {

	static final Validator validator = Validation.buildDefaultValidatorFactory().getValidator();

	static Set<String> invalidFields(Object request) {
		Set<? extends ConstraintViolation<?>> violations = validator.validate(request);
		return violations.stream().map(v -> v.getPropertyPath().toString()).collect(Collectors.toSet());
	}

	@Test
	void validRegistration() {
		assertThat(invalidFields(new RegisterRequest("developer", "developer@example.com", "example-password")))
			.isEmpty();
	}

	@ParameterizedTest
	@ValueSource(strings = { "", "ab", "-starts-with-dash", "has space", "way-too-long-username-over-32-chars",
			"emoji😀", "semi;colon" })
	void badUsernames(String username) {
		assertThat(invalidFields(new RegisterRequest(username, "a@example.com", "example-password")))
			.containsExactly("username");
	}

	@ParameterizedTest
	@ValueSource(strings = { "", "not-an-email", "a@", "@example.com" })
	void badEmails(String email) {
		assertThat(invalidFields(new RegisterRequest("developer", email, "example-password")))
			.containsExactly("email");
	}

	@Test
	void emailLengthIsBounded() {
		String email = "a".repeat(250) + "@example.com";
		assertThat(invalidFields(new RegisterRequest("developer", email, "example-password"))).contains("email");
	}

	@Test
	void passwordLengthIsBoundedBothWays() {
		assertThat(invalidFields(new RegisterRequest("developer", "a@example.com", "short"))).containsExactly("password");
		assertThat(invalidFields(new RegisterRequest("developer", "a@example.com", "x".repeat(129))))
			.containsExactly("password");
		assertThat(invalidFields(new RegisterRequest("developer", "a@example.com", null))).containsExactly("password");
		assertThat(invalidFields(new RegisterRequest("developer", "a@example.com", "x".repeat(128)))).isEmpty();
	}

	@Test
	void loginChecksPresenceAndSizeOnly() {
		assertThat(invalidFields(new LoginRequest("not-an-email-but-fine", "anything"))).isEmpty();
		assertThat(invalidFields(new LoginRequest("", "anything"))).containsExactly("email");
		assertThat(invalidFields(new LoginRequest("a@example.com", "x".repeat(129)))).containsExactly("password");
	}

	@Test
	void toStringNeverShowsThePassword() {
		assertThat(new RegisterRequest("u", "e", "hunter2-secret").toString()).doesNotContain("hunter2-secret");
		assertThat(new LoginRequest("e", "hunter2-secret").toString()).doesNotContain("hunter2-secret");
	}

	@Test
	void emailIsNormalised() {
		assertThat(AuthService.normaliseEmail("  Developer@Example.COM ")).isEqualTo("developer@example.com");
	}
}
