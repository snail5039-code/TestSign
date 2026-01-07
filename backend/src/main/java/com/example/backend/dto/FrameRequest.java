package com.example.backend.dto;

import java.util.List;
import lombok.Data;

@Data
public class FrameRequest {
    private List<Double> pose;      // 99
    private List<Double> face;      // (옵션) 지금은 0 배열로 받는 형태 유지 가능
    private List<Double> leftHand;  // 63
    private List<Double> rightHand; // 63
}