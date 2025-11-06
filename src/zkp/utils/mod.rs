#[cfg(feature = "zkp")]
pub mod fl2sa;

#[cfg(feature = "zkp")]
pub mod sparse;

#[cfg(feature = "zkp")]
pub mod mulfp;

#[cfg(feature = "zkp")]
pub mod linear;

#[cfg(feature = "zkp")]
pub mod sa2fl;

#[cfg(not(feature = "zkp"))]
pub mod fl2sa {

    #[derive(Debug, Clone, Copy)]
    pub enum FloatPrecision {
        Half,   // f16
        Single, // f32
        Double, // f64
    }

    #[derive(Debug, Clone)]
    pub struct FL2SAInput {
        pub precision: FloatPrecision,
        pub values: Vec<f32>,
    }

    #[derive(Debug, Clone)]
    pub struct FL2SAOutput {
        pub success: bool,
        pub error_msg: Option<String>,
    }

    pub fn fl2sa_convert(_input: FL2SAInput) -> FL2SAOutput {
        FL2SAOutput {
            success: false,
            error_msg: Some(
                "ZKP feature is not enabled; compile with --features zkp to enable support"
                    .to_string(),
            ),
        }
    }
}

#[cfg(not(feature = "zkp"))]
pub mod mulfp {

    #[derive(Debug, Clone)]
    pub struct MULFPResult {
        pub computed: bool,
        pub error: Option<String>,
    }

    pub fn mulfp_process(_data: Vec<f64>) -> MULFPResult {
        MULFPResult {
            computed: false,
            error: Some(
                "ZKP feature is not enabled; compile with --features zkp to enable support"
                    .to_string(),
            ),
        }
    }
}

#[cfg(not(feature = "zkp"))]
pub mod sa2fl {

    #[derive(Debug, Clone)]
    pub struct SA2FLOutput {
        pub success: bool,
        pub result: Option<Vec<f64>>,
        pub error: Option<String>,
    }

    pub fn sa2fl_process(_data: Vec<u64>) -> SA2FLOutput {
        SA2FLOutput {
            success: false,
            result: None,
            error: Some(
                "ZKP feature is not enabled; compile with --features zkp to enable support"
                    .to_string(),
            ),
        }
    }
}
