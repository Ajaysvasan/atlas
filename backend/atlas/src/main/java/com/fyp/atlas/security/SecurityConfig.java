package com.fyp.atlas.security;

import java.util.List;
import java.util.Set;

import com.fyp.atlas.config.AtlasProperties;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.oauth2.server.resource.web.BearerTokenResolver;
import org.springframework.security.oauth2.server.resource.web.DefaultBearerTokenResolver;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

/**
 * Stateless bearer-token security. Only registration, login and the health check are open; every
 * other route needs a valid access token.
 */
@Configuration(proxyBeanMethods = false)
public class SecurityConfig {

	static final Set<String> PUBLIC_POSTS = Set.of("/register", "/login");

	@Bean
	SecurityFilterChain api(HttpSecurity http, SecurityErrorHandlers handlers) throws Exception {
		return http
			.csrf(AbstractHttpConfigurer::disable)
			.cors(cors -> {
			})
			.httpBasic(AbstractHttpConfigurer::disable)
			.formLogin(AbstractHttpConfigurer::disable)
			.logout(AbstractHttpConfigurer::disable)
			.requestCache(AbstractHttpConfigurer::disable)
			.sessionManagement(session -> session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
			.authorizeHttpRequests(routes -> routes
				.requestMatchers(HttpMethod.POST, "/register", "/login").permitAll()
				.requestMatchers(HttpMethod.GET, "/actuator/health", "/actuator/health/**").permitAll()
				.requestMatchers(HttpMethod.OPTIONS, "/**").permitAll()
				.requestMatchers("/error").permitAll()
				.anyRequest().authenticated())
			.oauth2ResourceServer(resource -> resource
				.jwt(jwt -> {
				})
				.bearerTokenResolver(bearerTokenResolver())
				.authenticationEntryPoint(handlers)
				.accessDeniedHandler(handlers))
			.exceptionHandling(exceptions -> exceptions
				.authenticationEntryPoint(handlers)
				.accessDeniedHandler(handlers))
			.build();
	}

	/** A stale token sent to /login or /register is ignored rather than failing the request. */
	static BearerTokenResolver bearerTokenResolver() {
		DefaultBearerTokenResolver standard = new DefaultBearerTokenResolver();
		return request -> PUBLIC_POSTS.contains(request.getRequestURI()) ? null : standard.resolve(request);
	}

	/** Explicit origins only. Tokens travel in a header, not a cookie, so credentials stay off. */
	@Bean
	CorsConfigurationSource corsConfigurationSource(AtlasProperties properties) {
		List<String> origins = properties.cors().allowedOrigins().stream()
			.map(String::trim)
			.filter(origin -> !origin.isEmpty() && !origin.equals("*"))
			.toList();
		CorsConfiguration cors = new CorsConfiguration();
		cors.setAllowedOrigins(origins);
		cors.setAllowedMethods(List.of("GET", "POST", "OPTIONS"));
		cors.setAllowedHeaders(List.of("Authorization", "Content-Type", "X-Request-Id"));
		cors.setExposedHeaders(List.of("X-Request-Id", "Retry-After"));
		cors.setAllowCredentials(false);
		cors.setMaxAge(3600L);
		UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
		source.registerCorsConfiguration("/**", cors);
		return source;
	}
}
