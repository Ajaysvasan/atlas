package com.fyp.atlas.repository;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.time.OffsetDateTime;
import java.util.Optional;
import java.util.UUID;

import com.fyp.atlas.entity.UserAccount;

import org.springframework.dao.DuplicateKeyException;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.stereotype.Repository;

/** user_table, through plain SQL. Uniqueness is the database's to decide, not a prior lookup's. */
@Repository
public class 	UserRepository {

	public static final String EMAIL_KEY = "user_table_email_key";
	public static final String USERNAME_KEY = "user_table_username_key";

	private static final String COLUMNS = "user_id, username, email, password_hash, created_at, updated_at, is_active";

	private final JdbcClient jdbc;

	public UserRepository(JdbcClient jdbc) {
		this.jdbc = jdbc;
	}

	/** Inserts the account; a DuplicateKeyException names the unique index it hit. */
	public UserAccount insert(UUID userId, String username, String email, String passwordHash)
			throws DuplicateKeyException {
		return jdbc.sql("insert into user_table (user_id, username, email, password_hash) values (?, ?, ?, ?) "
				+ "returning " + COLUMNS)
			.params(userId, username, email, passwordHash)
			.query(UserRepository::map)
			.single();
	}

	public Optional<UserAccount> findByEmail(String normalisedEmail) {
		return jdbc.sql("select " + COLUMNS + " from user_table where email = ?")
			.param(normalisedEmail)
			.query(UserRepository::map)
			.optional();
	}

	public Optional<UserAccount> findById(UUID userId) {
		return jdbc.sql("select " + COLUMNS + " from user_table where user_id = ?")
			.param(userId)
			.query(UserRepository::map)
			.optional();
	}

	public boolean existsByEmail(String normalisedEmail) {
		return jdbc.sql("select exists(select 1 from user_table where email = ?)")
			.param(normalisedEmail)
			.query(Boolean.class)
			.single();
	}

	public boolean existsByUsername(String username) {
		return jdbc.sql("select exists(select 1 from user_table where lower(username) = lower(?))")
			.param(username)
			.query(Boolean.class)
			.single();
	}

	public boolean isActive(UUID userId) {
		return jdbc.sql("select coalesce((select is_active from user_table where user_id = ?), false)")
			.param(userId)
			.query(Boolean.class)
			.single();
	}

	/** Replaces the hash only if it is still the one that was verified, so a concurrent change wins. */
	public boolean replacePasswordHash(UUID userId, String expectedHash, String newHash) {
		return jdbc.sql("update user_table set password_hash = ?, updated_at = now() "
				+ "where user_id = ? and password_hash = ?")
			.params(newHash, userId, expectedHash)
			.update() == 1;
	}

	private static UserAccount map(ResultSet row, int index) throws SQLException {
		return new UserAccount(row.getObject("user_id", UUID.class), row.getString("username"),
				row.getString("email"), row.getString("password_hash"),
				row.getObject("created_at", OffsetDateTime.class), row.getObject("updated_at", OffsetDateTime.class),
				row.getBoolean("is_active"));
	}
}
