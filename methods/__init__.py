from methods.adaptive_stability import AdaptiveStabilityDetection
from methods.agentauditor_style import AgentAuditorStyle
from methods.c3_style import C3StyleCausalCreditAnalysis
from methods.cider import CiderFull, CiderFullTuned, CiderSOTA, CiderVerified
from methods.cider_v2 import CiderAdaptiveGated
from methods.conformal_social_choice import ConformalSocialChoice
from methods.consensagent_style import ConsensAgentStyle
from methods.dar_style import DARStyleDiversityAwareRetention
from methods.free_mad_style import FreeMADStyle
from methods.majority_vote import MajorityVote
from methods.official_adapters import DAROfficialAdapter, FreeMADOfficialAdapter
from methods.self_consistency import SelfConsistency
from methods.single_agent import SingleAgent
from methods.standard_mad import StandardMultiAgentDebate

METHODS = {
    "single_agent": SingleAgent,
    "self_consistency": SelfConsistency,
    "majority_vote": MajorityVote,
    "standard_multi_agent_debate": StandardMultiAgentDebate,
    "consensagent_style": ConsensAgentStyle,
    "free_mad_style": FreeMADStyle,
    "agentauditor_style": AgentAuditorStyle,
    "c3_style_causal_credit_analysis": C3StyleCausalCreditAnalysis,
    "conformal_social_choice": ConformalSocialChoice,
    "dar_style_diversity_aware_retention": DARStyleDiversityAwareRetention,
    "adaptive_stability_detection": AdaptiveStabilityDetection,
    "cider_full": CiderFull,
    "cider_full_tuned": CiderFullTuned,
    "cider_verified": CiderVerified,
    "cider_sota": CiderSOTA,
    "cider_adaptive_gated": CiderAdaptiveGated,
    "free_mad_official_adapter": FreeMADOfficialAdapter,
    "dar_official_adapter": DAROfficialAdapter,
}
