package com.example.backend.controller;


import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClient;

@RestController
@RequestMapping("/api")
public class TranslateController {
	
    private final WebClient webClient;

    public TranslateController(
            WebClient.Builder builder,
            @Value("${fastapi.base-url}") String baseUrl
    ) {
        this.webClient = builder.baseUrl(baseUrl).build();
    }
    
    private static final ParameterizedTypeReference<Map<String, Object>> MAP_STRING_OBJECT =
            new ParameterizedTypeReference<>() {};
            
    @PostMapping(
            value = "/translate",
            consumes = MediaType.APPLICATION_JSON_VALUE,
            produces = MediaType.APPLICATION_JSON_VALUE
    )
    public Map<String, Object> translate(@RequestBody Map<String, Object> body) {

        // 프론트: { frames: [...] }
        // FastAPI: /predict 가 { frames: [...] } 받도록 되어있으니 그대로 전달
        Map<String, Object> fast = webClient.post()
                .uri("/predict")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(MAP_STRING_OBJECT)
                .block();

        if (fast == null) {
            return Map.of(
                    "label", "unknown",
                    "confidence", 0.0,
                    "text", "unknown",
                    "raw", Map.of()
            );
        }

        // 프론트 UI가 label/confidence/text를 기대하는 형태라면 여기서 맞춰줌
        Object label = fast.getOrDefault("label", fast.get("pred"));
        Object confidence = fast.getOrDefault("confidence", fast.getOrDefault("score", 0.0));

        fast.put("label", label);
        fast.put("confidence", confidence);
        fast.putIfAbsent("text", label);
        
        if (!fast.containsKey("candidates")) {
            Object top5 = fast.get("top5");
            if (top5 != null) fast.put("candidates", top5);
        }

        return fast;
    }
}