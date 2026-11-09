package com.fyp.atlas.grpc;

import java.io.File;
import java.util.concurrent.TimeUnit;

import javax.net.ssl.SSLException;

import com.fyp.atlas.config.AtlasProperties;
import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslContextBuilder;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/** One long-lived channel to the Python pipeline: plaintext on loopback, TLS (optionally mTLS) otherwise. */
@Configuration(proxyBeanMethods = false)
public class GrpcChannelConfig {

	private static final Logger log = LoggerFactory.getLogger(GrpcChannelConfig.class);

	static final int MIN_TOKEN_LENGTH = 16;

	@Bean(destroyMethod = "")
	ManagedChannel atlasChannel(AtlasProperties properties) throws SSLException {
		AtlasProperties.Grpc grpc = properties.grpc();
		if (grpc.serviceToken() == null || grpc.serviceToken().length() < MIN_TOKEN_LENGTH) {
			throw new IllegalStateException("ATLAS_GRPC_SERVICE_TOKEN must be set (at least " + MIN_TOKEN_LENGTH
					+ " characters), to the value the Python gRPC server uses");
		}
		NettyChannelBuilder builder = NettyChannelBuilder.forAddress(grpc.host(), grpc.port())
			.maxInboundMessageSize((int) grpc.maxInboundMessageSize().toBytes())
			.keepAliveTime(5, TimeUnit.MINUTES)
			.keepAliveWithoutCalls(false);
		AtlasProperties.Tls tls = grpc.tls();
		if (tls.enabled()) {
			SslContextBuilder ssl = GrpcSslContexts.forClient();
			if (notBlank(tls.trustCertCollection())) {
				ssl.trustManager(new File(tls.trustCertCollection()));
			}
			if (notBlank(tls.clientCertChain()) && notBlank(tls.clientPrivateKey())) {
				ssl.keyManager(new File(tls.clientCertChain()), new File(tls.clientPrivateKey()));
			}
			builder.sslContext(ssl.build()).useTransportSecurity();
			if (notBlank(tls.authorityOverride())) {
				builder.overrideAuthority(tls.authorityOverride());
			}
		}
		else {
			if (!isLoopback(grpc.host())) {
				log.warn("gRPC to {} is plaintext; enable atlas.grpc.tls for anything but loopback", grpc.host());
			}
			builder.usePlaintext();
		}
		log.info("Atlas pipeline channel: {}:{} ({})", grpc.host(), grpc.port(), tls.enabled() ? "TLS" : "plaintext");
		return builder.build();
	}

	@Bean
	ChannelLifecycle atlasChannelLifecycle(ManagedChannel atlasChannel) {
		return new ChannelLifecycle(atlasChannel);
	}

	static boolean notBlank(String value) {
		return value != null && !value.isBlank();
	}

	static boolean isLoopback(String host) {
		return "127.0.0.1".equals(host) || "localhost".equals(host) || "::1".equals(host);
	}

	/** Drains in-flight calls on shutdown, then forces the channel closed. */
	public static final class ChannelLifecycle implements AutoCloseable {

		private final ManagedChannel channel;

		ChannelLifecycle(ManagedChannel channel) {
			this.channel = channel;
		}

		@Override
		public void close() throws InterruptedException {
			channel.shutdown();
			if (!channel.awaitTermination(10, TimeUnit.SECONDS)) {
				channel.shutdownNow();
			}
		}
	}
}
