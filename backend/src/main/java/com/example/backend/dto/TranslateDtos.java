package com.example.backend.dto;

import jakarta.validation.constraints.NotNull;
import java.util.List;

public class TranslateDtos {
  public static class Frame {
    public List<Object> pose;
    public List<Object> leftHand;
    public List<Object> rightHand;
    public List<Object> face;
  }

  public static class TranslateRequest {
    @NotNull public List<Frame> frames;
  }

  public static class TranslateResponse {
    public String label;
    public String text;
    public double confidence;
    public List<List<Object>> candidates;
  }
}
