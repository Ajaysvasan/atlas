package com.fyp.atlas.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public record RegisterRequest(
		@NotBlank(message = "is required")
		@Pattern(regexp = "^[A-Za-z0-9][A-Za-z0-9_.-]{2,31}$",
				message = "must be 3-32 characters: letters, digits, '.', '_' or '-', starting with a letter or digit")
		String username,

		@NotBlank(message = "is required")
		@Size(max = 254, message = "must be at most 254 characters")
		@Email(message = "must be a valid email address")
		String email,

		@NotNull(message = "is required")
		@Size(min = 8, max = 128, message = "must be 8-128 characters")
		String password) {

	/** Surrounding whitespace is a client artefact; the password is kept exactly as typed. */
	public RegisterRequest {
		username = username == null ? null : username.trim();
		email = email == null ? null : email.trim();
	}

	@Override
	public String toString() {
		return "RegisterRequest[username=" + username + ", email=" + email + ", password=<redacted>]";
	}
}
