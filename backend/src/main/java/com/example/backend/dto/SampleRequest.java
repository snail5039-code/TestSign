package com.example.backend.dto;

import java.util.List;

import lombok.Data;

@Data
public class SampleRequest {
    private int version;
    private String createdAt;
    private String label;
    private String labelKo;
    private String note;
    private List<FrameRequest> frames;
}
