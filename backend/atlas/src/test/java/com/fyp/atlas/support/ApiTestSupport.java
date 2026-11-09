package com.fyp.atlas.support;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;

import tools.jackson.databind.JsonNode;
import tools.jackson.databind.json.JsonMapper;

import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;
import org.springframework.test.web.servlet.request.AbstractMockHttpServletRequestBuilder;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;

/** Register, log in and call with a token, each from its own client address so rate limits stay out of the way. */
public final class ApiTestSupport {

	private static final AtomicInteger ADDRESS = new AtomicInteger(1);

	private final MockMvc mvc;
	private final JsonMapper json;

	public ApiTestSupport(MockMvc mvc, JsonMapper json) {
		this.mvc = mvc;
		this.json = json;
	}

	public static <B extends AbstractMockHttpServletRequestBuilder<B>> B from(B builder, String address) {
		return builder.with(request -> {
			request.setRemoteAddr(address);
			return request;
		});
	}

	public static String freshAddress() {
		int n = ADDRESS.getAndIncrement();
		return "10." + ((n >> 16) & 255) + "." + ((n >> 8) & 255) + "." + (n & 255);
	}

	public MockHttpServletRequestBuilder jsonPost(String path, Object body) {
		return from(post(path).contentType(MediaType.APPLICATION_JSON).content(json.writeValueAsString(body)),
				freshAddress());
	}

	public JsonNode body(MvcResult result) throws Exception {
		return json.readTree(result.getResponse().getContentAsString());
	}

	public record Account(String userId, String username, String email, String password, String token) {
	}

	public Account registerAndLogin() throws Exception {
		String suffix = UUID.randomUUID().toString().substring(0, 8);
		String username = "user_" + suffix;
		String email = "user." + suffix + "@example.com";
		String password = "password-" + suffix;
		MvcResult registered = mvc
			.perform(jsonPost("/register", Map.of("username", username, "email", email, "password", password)))
			.andReturn();
		if (registered.getResponse().getStatus() != 201) {
			throw new IllegalStateException("register: " + registered.getResponse().getContentAsString());
		}
		String userId = body(registered).get("userId").asString();
		MvcResult login = mvc.perform(jsonPost("/login", Map.of("email", email, "password", password))).andReturn();
		String token = body(login).get("accessToken").asString();
		return new Account(userId, username, email, password, token);
	}
}
