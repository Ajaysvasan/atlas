package com.fyp.atlas.support;

import java.io.IOException;

import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Primary;

/** The application's channel points at an in-process FakePipeline instead of Python. */
@TestConfiguration(proxyBeanMethods = false)
public class FakePipelineConfig {

	static final String NAME = "fake-atlas-pipeline";

	@Bean
	FakePipeline fakePipeline() {
		return new FakePipeline();
	}

	@Bean(destroyMethod = "shutdownNow")
	Server fakePipelineServer(FakePipeline fake, @Value("${atlas.grpc.service-token}") String token)
			throws IOException {
		return InProcessServerBuilder.forName(NAME)
			.addService(fake)
			.intercept(fake.tokenCheck(token))
			.build()
			.start();
	}

	@Bean
	@Primary
	ManagedChannel fakePipelineChannel(Server fakePipelineServer) {
		return InProcessChannelBuilder.forName(NAME).build();
	}
}
