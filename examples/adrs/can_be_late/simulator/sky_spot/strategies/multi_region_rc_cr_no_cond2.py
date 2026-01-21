"""
Our policy: Uniform Progress without condition2 (no sticky ON_DEMAND).
Multi-region version - finds ANY region with spot availability.
"""

import logging

from sky_spot.strategies.multi_region_rc_cr_threshold import MultiRegionRCCRThresholdStrategy

logger = logging.getLogger(__name__)


class MultiRegionRCCRNoCondition2Strategy(MultiRegionRCCRThresholdStrategy):
    NAME = "multi_region_rc_cr_no_cond2"
    
    def _condition2(self):
        """Override to always return negative, disabling sticky ON_DEMAND behavior."""
        return -1.0
