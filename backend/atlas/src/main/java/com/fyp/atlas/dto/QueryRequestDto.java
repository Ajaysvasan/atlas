package com.fyp.atlas.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

/**
 * POST /query. There is no user field: the user is the token's subject. Unknown JSON properties
 * (such as a client-supplied "userId") are ignored, never trusted.
 */
public record QueryRequestDto(
		@Pattern(regexp = "^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$", message = "is not a conversation id")
		String conversationId,

		@NotBlank(message = "is required")
		@Size(max = 8000, message = "must be at most 8000 characters")
		String query) {
}
