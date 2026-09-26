#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PredictiveRegime {
    Proxy,
    Lookahead,
    EscalatedExact,
    Exact,
    Finite,
}

impl PredictiveRegime {
    pub fn label(self) -> &'static str {
        match self {
            Self::Proxy => "proxy",
            Self::Lookahead => "lookahead",
            Self::EscalatedExact => "escalated_exact",
            Self::Exact => "exact",
            Self::Finite => "finite",
        }
    }
}
