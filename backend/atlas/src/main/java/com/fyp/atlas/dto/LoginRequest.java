package com.fyp.atlas.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

/** Only presence and size are checked: a format error would tell a caller which emails can exist. */
public record LoginRequest(
		@NotBlank(message = "is required") @Size(max = 254, message = "must be at most 254 characters") String email,
		@NotNull(message = "is required") @Size(max = 128, message = "must be at most 128 characters") String password) {

	@Override
	public String toString() {
		return "LoginRequest[email=" + email + ", password=<redacted>]";
	}
}
