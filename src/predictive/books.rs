#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PredictivePromotionSource {
    ExactDateOpenerArtifact,
    RecentOpenerArtifact,
    ReplyBook,
    RecentReplyBook,
    SessionRootFallback,
    SessionReplyFallback,
    SessionThirdFallback,
}
