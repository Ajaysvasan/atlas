package com.fyp.atlas.entity;

import java.time.OffsetDateTime;
import java.util.UUID;

/** One user_table row. The hash never leaves the service layer. */
public record UserAccount(
		UUID userId,
		String username,
		String email,
		String passwordHash,
		OffsetDateTime createdAt,
		OffsetDateTime updatedAt,
		boolean active) {

	@Override
	public String toString() {
		return "UserAccount[userId=" + userId + ", username=" + username + ", active=" + active + "]";
	}
}
