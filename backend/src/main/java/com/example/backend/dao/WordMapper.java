package com.example.backend.dao;

import org.apache.ibatis.annotations.*;
import java.util.Optional;

@Mapper
public interface WordMapper {
  @Select("SELECT ko_text FROM word WHERE label = #{label}")
  Optional<String> findKoText(@Param("label") String label);

  @Insert("""
    INSERT INTO translate_log(label, ko_text, confidence)
    VALUES(#{label}, #{koText}, #{confidence})
  """)
  void insertLog(@Param("label") String label,
                 @Param("koText") String koText,
                 @Param("confidence") double confidence);
}