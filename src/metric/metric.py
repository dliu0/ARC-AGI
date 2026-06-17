def arc_grid_accuracy(predictions: list, task_data: dict) -> float:
    """
    Evaluates the accuracy of the ARC-AGI predictions.
    Each task has one or more 'test' pairs.
    A prediction is correct only if it exactly matches the target output grid.
    Returns 1.0 if all test predictions match perfectly, else 0.0.
    """
    test_cases = task_data.get("test", [])
    if not test_cases or len(predictions) != len(test_cases):
        return 0.0

    for pred, test_case in zip(predictions, test_cases):
        target = test_case.get("output", [])
        if pred != target:
            return 0.0
            
    return 1.0
