package com.example.backend.service;

import java.time.Duration;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import com.example.backend.dto.TranslateDtos.TranslateRequest;
import com.example.backend.dto.TranslateDtos.TranslateResponse;

@Service
public class PythonTranslateService {
  private final WebClient web;

  public PythonTranslateService(@Value("${app.python.base-url}") String baseUrl) {
    this.web = WebClient.builder().baseUrl(baseUrl).build();
  }

  public TranslateResponse predict(TranslateRequest req) {
    return web.post()
      .uri("/predict")
      .bodyValue(req)
      .retrieve()
      .bodyToMono(TranslateResponse.class)
      .timeout(Duration.ofSeconds(5))
      .block();
  }
}
