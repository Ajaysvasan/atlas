package com.fyp.atlas.repository;

import static org.assertj.core.api.Assertions.assertThat;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;

import com.fyp.atlas.support.PostgresContainer;

import org.flywaydb.core.Flyway;
import org.flywaydb.core.api.output.MigrateResult;
import org.junit.jupiter.api.Test;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.postgresql.PostgreSQLContainer;

/** The migration against an empty database and against one that already holds the Python side's tables. */
@Testcontainers
class MigrationTest {

	@Container
	static final PostgreSQLContainer postgres = new PostgreSQLContainer(PostgresContainer.IMAGE);

	/** The same settings application.properties gives Flyway. */
	static Flyway flyway(String database) {
		return Flyway.configure()
			.dataSource(url(database), postgres.getUsername(), postgres.getPassword())
			.locations("classpath:db/migration")
			.baselineOnMigrate(true)
			.baselineVersion("0")
			.cleanDisabled(true)
			.table("atlas_flyway_history")
			.load();
	}

	static String url(String database) {
		return postgres.getJdbcUrl().replaceFirst("/[^/?]+(\\?|$)", "/" + database + "$1");
	}

	static void createDatabase(String name) throws Exception {
		try (Connection c = DriverManager.getConnection(postgres.getJdbcUrl(), postgres.getUsername(),
				postgres.getPassword()); Statement s = c.createStatement()) {
			s.execute("create database " + name);
		}
	}

	static List<String> query(String database, String sql) throws Exception {
		List<String> rows = new ArrayList<>();
		try (Connection c = DriverManager.getConnection(url(database), postgres.getUsername(), postgres.getPassword());
				Statement s = c.createStatement(); ResultSet r = s.executeQuery(sql)) {
			while (r.next()) {
				rows.add(r.getString(1));
			}
		}
		return rows;
	}

	@Test
	void cleanDatabaseGetsUserTableWithItsConstraints() throws Exception {
		createDatabase("clean_db");
		MigrateResult result = flyway("clean_db").migrate();
		assertThat(result.success).isTrue();
		assertThat(result.migrationsExecuted).isEqualTo(1);

		assertThat(query("clean_db", "select column_name || ':' || data_type || ':' || is_nullable "
				+ "from information_schema.columns where table_name = 'user_table' order by ordinal_position"))
			.containsExactly("user_id:uuid:NO", "username:character varying:NO", "email:character varying:NO",
					"password_hash:character varying:NO", "created_at:timestamp with time zone:NO",
					"updated_at:timestamp with time zone:NO", "is_active:boolean:NO");
		assertThat(query("clean_db", "select indexname from pg_indexes where tablename = 'user_table' order by 1"))
			.containsExactly("user_table_email_key", "user_table_pkey", "user_table_username_key");
		assertThat(query("clean_db",
				"select conname from pg_constraint where conrelid = 'user_table'::regclass and contype = 'c' order by 1"))
			.containsExactly("user_table_email_normalised", "user_table_password_hash_encoded",
					"user_table_username_format");

		assertThat(flyway("clean_db").migrate().migrationsExecuted).isZero();
	}

	@Test
	void existingAtlasTablesAndRowsSurvive() throws Exception {
		createDatabase("atlas_existing");
		try (Connection c = DriverManager.getConnection(url("atlas_existing"), postgres.getUsername(),
				postgres.getPassword()); Statement s = c.createStatement()) {
			s.execute("create table conversations(conversation_id varchar primary key, user_id varchar not null, "
					+ "conversation_name varchar not null, created_at timestamptz not null)");
			s.execute("insert into conversations values ('c1', 'ajay', 'PostgreSQL MVCC', now())");
			s.execute("create table conversation_table(exchange_id varchar primary key, query text)");
			s.execute("insert into conversation_table values ('e1', 'what is mvcc')");
			s.execute("create table vectors(project_id varchar, vector_id bigint, primary key (project_id, vector_id))");
			s.execute("insert into vectors values ('global', 1)");
		}

		MigrateResult result = flyway("atlas_existing").migrate();
		assertThat(result.success).isTrue();

		assertThat(query("atlas_existing", "select conversation_name from conversations")).containsExactly("PostgreSQL MVCC");
		assertThat(query("atlas_existing", "select query from conversation_table")).containsExactly("what is mvcc");
		assertThat(query("atlas_existing", "select count(*) from vectors")).containsExactly("1");
		assertThat(query("atlas_existing", "select to_regclass('user_table')::text")).containsExactly("user_table");
		assertThat(query("atlas_existing", "select version from atlas_flyway_history where success order by installed_rank"))
			.containsExactly("0", "1");
	}
}
