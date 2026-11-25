"""Hidden ground truth store and the dynamic test protocol (Sections 4.1, 11.1).

Only the harness and the evaluator read the ground truth. The estimator and the
planner never import this package (see ``guard.assert_no_truth_imports``).
"""

from .truth import GroundTruthStore  # noqa: F401
from .tests import TestOutcome, tilt_test, shake_test, push_test, fixed_perturbation, failure_check, run_protocol  # noqa: F401
