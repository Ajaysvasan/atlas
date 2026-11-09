package com.fyp.atlas.controller;

import com.fyp.atlas.dto.ConversationListDto;
import com.fyp.atlas.dto.ExchangeListDto;
import com.fyp.atlas.security.AuthenticatedUser;
import com.fyp.atlas.service.ConversationService;
import com.fyp.atlas.web.RequestCorrelation;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/** The caller's conversations and their exchanges; the user is the token's subject, never a parameter. */
@RestController
public class ConversationController {

	private final ConversationService conversations;

	public ConversationController(ConversationService conversations) {
		this.conversations = conversations;
	}

	/** limit is read as text so a malformed value is a 400 naming the field, not a 500. */
	@GetMapping("/conversations")
	public ConversationListDto list(@RequestParam(required = false) String limit,
			@RequestParam(required = false) String cursor, Authentication authentication, HttpServletRequest request) {
		return conversations.list(AuthenticatedUser.id(authentication), limit, cursor, RequestCorrelation.of(request));
	}

	@GetMapping("/conversations/{conversationId}/exchanges")
	public ExchangeListDto exchanges(@PathVariable String conversationId, @RequestParam(required = false) String limit,
			@RequestParam(required = false) String cursor, Authentication authentication, HttpServletRequest request) {
		return conversations.exchanges(AuthenticatedUser.id(authentication), conversationId, limit, cursor,
				RequestCorrelation.of(request));
	}
}
