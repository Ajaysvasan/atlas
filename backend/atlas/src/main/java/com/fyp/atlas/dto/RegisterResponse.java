package com.fyp.atlas.dto;

import java.time.OffsetDateTime;
import java.util.UUID;

/** What registration returns: never the password or its hash. */
public record RegisterResponse(UUID userId, String username, String email, OffsetDateTime createdAt) {
}
