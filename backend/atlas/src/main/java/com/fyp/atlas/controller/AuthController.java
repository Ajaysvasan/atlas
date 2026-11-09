package com.fyp.atlas.controller;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;

import com.fyp.atlas.dto.LoginRequest;
import com.fyp.atlas.dto.RegisterRequest;
import com.fyp.atlas.dto.RegisterResponse;
import com.fyp.atlas.dto.TokenResponse;
import com.fyp.atlas.service.AuthService;

import org.springframework.http.CacheControl;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class AuthController {

	private final AuthService auth;

	public AuthController(AuthService auth) {
		this.auth = auth;
	}

	@PostMapping(path = "/register", consumes = MediaType.APPLICATION_JSON_VALUE)
	public ResponseEntity<RegisterResponse> register(@Valid @RequestBody RegisterRequest body,
			HttpServletRequest request) {
		return ResponseEntity.status(HttpStatus.CREATED).body(auth.register(body, request.getRemoteAddr()));
	}

	@PostMapping(path = "/login", consumes = MediaType.APPLICATION_JSON_VALUE)
	public ResponseEntity<TokenResponse> login(@Valid @RequestBody LoginRequest body, HttpServletRequest request) {
		return ResponseEntity.ok()
			.cacheControl(CacheControl.noStore())
			.body(auth.login(body, request.getRemoteAddr()));
	}
}
