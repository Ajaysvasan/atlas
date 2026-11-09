package com.fyp.atlas.support;

import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.boot.testcontainers.service.connection.ServiceConnection;
import org.springframework.context.annotation.Bean;
import org.testcontainers.postgresql.PostgreSQLContainer;

/** One throwaway PostgreSQL per test JVM; the application's datasource points at it. */
@TestConfiguration(proxyBeanMethods = false)
public class PostgresContainer {

	public static final String IMAGE = "postgres:17-alpine";

	@Bean
	@ServiceConnection
	PostgreSQLContainer postgres() {
		return new PostgreSQLContainer(IMAGE).withReuse(false);
	}
}
