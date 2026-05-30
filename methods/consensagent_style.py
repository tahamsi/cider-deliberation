from __future__ import annotations

from methods.standard_mad import StandardMultiAgentDebate


class ConsensAgentStyle(StandardMultiAgentDebate):
    name = "consensagent_style"
    deviation_note = "style approximation: consensus-seeking debate using majority over final visible round"
