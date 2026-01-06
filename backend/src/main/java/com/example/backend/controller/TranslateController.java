package com.example.backend.controller;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.example.backend.dao.WordMapper;
import com.example.backend.dto.TranslateDtos.TranslateRequest;
import com.example.backend.dto.TranslateDtos.TranslateResponse;
import com.example.backend.service.PythonTranslateService;

import jakarta.validation.Valid;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = {"http://localhost:5173"})
public class TranslateController {
  private final PythonTranslateService py;
  private final WordMapper wordMapper;

  public TranslateController(PythonTranslateService py, WordMapper wordMapper) {
    this.py = py;
    this.wordMapper = wordMapper;
  }

  @PostMapping("/translate")
  public TranslateResponse translate(@Valid @RequestBody TranslateRequest req) {
    TranslateResponse res = py.predict(req);

    // DB에 word 매핑 있으면 text 덮기(권장)
    wordMapper.findKoText(res.label).ifPresent(v -> res.text = v);

    wordMapper.insertLog(res.label, res.text, res.confidence);
    return res;
  }
}