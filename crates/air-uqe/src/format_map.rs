#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FamilyId {
    F1 = 1,
    F2 = 2,
    F3 = 3,
    F4 = 4,
    F5 = 5,
    F6 = 6,
}

pub fn map_format(name: &str) -> FamilyId {
    match name {
        "Q2_K" | "Q3_K_S" | "Q3_K_M" | "Q3_K_L" | "Q4_0" | "Q4_1"
        | "Q4_K_S" | "Q4_K_M" | "Q4_K_L" | "Q5_0" | "Q5_1"
        | "Q5_K_S" | "Q5_K_M" | "Q5_K_L" | "Q6_K" | "Q8_0" | "Q8_K"
        | "FP16" | "BF16"
        | "UD-Q2_K_S" | "UD-Q2_K" | "UD-Q2_K_L" | "UD-Q2_K_XL"
        | "UD-Q3_K_S" | "UD-Q3_K_M" | "UD-Q3_K_L" | "UD-Q3_K_XL"
        | "UD-Q4_0" | "UD-Q4_1" | "UD-Q4_K_S" | "UD-Q4_K_M" | "UD-Q4_K_L" | "UD-Q4_XL"
        | "UD-Q5_0" | "UD-Q5_1" | "UD-Q5_K_S" | "UD-Q5_K_M" | "UD-Q5_K_L" | "UD-Q5_XL"
        | "UD-Q6_K_S" | "UD-Q6_K_M" | "UD-Q6_K_L" | "UD-Q6_XL"
        | "UD-Q8_0" | "UD-Q8_K_S" | "UD-Q8_K_M" | "UD-Q8_K_L" | "UD-Q8_XL" => FamilyId::F1,

        "IQ1_S" | "IQ1_M" | "IQ2_XXS" | "IQ2_XS" | "IQ2_S" | "IQ2_M"
        | "IQ3_XXS" | "IQ3_XS" | "IQ3_S" | "IQ3_M" | "IQ4_XS" | "IQ4_NL"
        | "UD-IQ1_S" | "UD-IQ1_M" | "UD-IQ2_XXS" | "UD-IQ2_XS" | "UD-IQ2_S" | "UD-IQ2_M"
        | "UD-IQ3_XXS" | "UD-IQ3_XS" | "UD-IQ3_S" | "UD-IQ3_M" | "UD-IQ4_XS" | "UD-IQ4_NL" => FamilyId::F2,

        "FP8-E4M3" | "FP8-E5M2" | "FP8-Dynamic" | "NVFP4" | "MXFP4" | "MXFP6" | "MXFP8" => FamilyId::F3,

        "LLM.int8()" | "W3A16-GPTQ" | "W4A16-GPTQ" | "W8A16-GPTQ"
        | "W4A16-AWQ" | "W8A16-AWQ" | "W8A8-SmoothQuant" | "W4A8-AutoRound" => FamilyId::F4,

        "BitNet-b1" | "BitNet-b1.58" | "NF4" | "DQ-NF4" | "Ternary" => FamilyId::F5,

        "EXL2" | "AQT" | "HQQ" => FamilyId::F6,

        _ => FamilyId::F1,
    }
}
