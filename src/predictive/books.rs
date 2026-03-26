#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PredictivePromotionSource {
    ExactDateOpenerArtifact,
    RecentOpenerArtifact,
    ReplyBook,
    SessionRootFallback,
    SessionReplyFallback,
    SessionThirdFallback,
}
