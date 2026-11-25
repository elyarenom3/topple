from topple.harness.guard import assert_no_truth_imports


def test_estimator_and_planner_never_import_ground_truth():
    assert assert_no_truth_imports() == []
