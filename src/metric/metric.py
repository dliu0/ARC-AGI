def _extract(args, kwargs):
    """Pull (test_cases, predictions) out of either DSPy-style positional
    args (example, pred, trace) or the keyword form the evaluator passes."""
    test_cases = None
    predictions = None

    if len(args) >= 2:
        example, pred = args[0], args[1]
        if hasattr(example, "test"):
            test_cases = example.test
        elif isinstance(example, dict):
            test_cases = example.get("test")
        if hasattr(pred, "output"):
            predictions = pred.output
        elif hasattr(pred, "predictions"):
            predictions = pred.predictions
        else:
            predictions = pred

    if test_cases is None:
        test_cases = kwargs.get("test")
    if predictions is None:
        predictions = (
            kwargs.get("predictions")
            or kwargs.get("pred")
            or kwargs.get("output")
        )
    return test_cases or [], predictions or []


def _dims(grid):
    if not isinstance(grid, list) or not grid:
        return (0, 0)
    if not all(isinstance(row, list) for row in grid):
        return (-1, -1)  # malformed (not a 2D list)
    width = len(grid[0])
    if not all(len(row) == width for row in grid):
        return (-2, -2)  # ragged
    return (len(grid), width)


def _cell_accuracy(pred, target):
    """Fraction of matching cells; assumes equal dimensions."""
    total = 0
    correct = 0
    for prow, trow in zip(pred, target):
        for pcell, tcell in zip(prow, trow):
            total += 1
            if pcell == tcell:
                correct += 1
    return (correct / total) if total else 0.0


def arc_grid_accuracy(*args, **kwargs):
    """Continuous ARC grid metric with textual feedback for the optimizer.

    Returns ``{"score": float, "feedback": str}``. The score is the mean over
    test cases of per-cell accuracy, gated on getting the output dimensions
    right (a dimension mismatch scores 0 for that case, since cell-by-cell
    comparison is undefined). A perfect exact match on every test case scores
    1.0 — preserving ARC's strict exact-solution requirement at the top end —
    while partial progress now yields a non-zero, monotonic signal instead of
    the old all-or-nothing 0/1. The feedback string explains, per case, what
    went wrong (parse/shape failures, dimension mismatches, how many cells were
    wrong) so the reflective optimizer has a gradient to act on.
    """
    test_cases, predictions = _extract(args, kwargs)

    if not test_cases:
        return {"score": 0.0, "feedback": "No test cases were available to score."}

    n = len(test_cases)
    case_scores = []
    notes = []
    exact = 0

    for i in range(n):
        target = test_cases[i].get("output", []) if isinstance(test_cases[i], dict) else []
        pred = predictions[i] if i < len(predictions) else None

        if pred is None:
            case_scores.append(0.0)
            notes.append(f"case {i}: no prediction returned")
            continue

        ph, pw = _dims(pred)
        th, tw = _dims(target)

        if ph < 0:
            case_scores.append(0.0)
            notes.append(f"case {i}: prediction is not a well-formed 2D grid")
            continue
        if (ph, pw) != (th, tw):
            case_scores.append(0.0)
            notes.append(
                f"case {i}: wrong output dimensions — predicted {ph}x{pw}, expected {th}x{tw}"
            )
            continue

        acc = _cell_accuracy(pred, target)
        case_scores.append(acc)
        if acc == 1.0:
            exact += 1
        else:
            wrong = round((1.0 - acc) * th * tw)
            notes.append(
                f"case {i}: correct shape ({th}x{tw}) but {wrong}/{th * tw} cells wrong "
                f"({acc * 100:.0f}% cell accuracy)"
            )

    score = sum(case_scores) / n  # each case_score already in [0, 1]

    summary = (
        f"Solved {exact}/{n} test case(s) exactly. "
        f"Mean cell accuracy {score * 100:.0f}%."
    )
    detail = " ".join(notes[:5])
    feedback = (summary + " " + detail).strip() if detail else summary

    return {"score": score, "feedback": feedback}
