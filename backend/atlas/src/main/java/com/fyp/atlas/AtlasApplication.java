package com.fyp.atlas;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class AtlasApplication {

	public static void main(String[] args) {
		SpringApplication.run(AtlasApplication.class, args);
	}

}
