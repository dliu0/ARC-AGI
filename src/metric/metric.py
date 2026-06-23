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


def _grid_str(grid):
    """Readable, one-row-per-line rendering of a grid for optimizer feedback.

    The raw nested-list repr is hard for the reflective LM to reason over; laying
    each row on its own line preserves the spatial pattern. Falls back to repr()
    for anything that is not a clean 2D list.
    """
    if not isinstance(grid, list) or not grid or not all(isinstance(r, list) for r in grid):
        return repr(grid)
    return "\n" + "\n".join(" ".join(str(c) for c in row) for row in grid)


def arc_grid_accuracy(*args, **kwargs):
    """Strict ARC grid metric with answer-revealing feedback (GEPA-style).

    Returns ``{"score": float, "feedback": str}``.

    SCORE is strictly 0.0 or 1.0 per test case (exact match required) — the real
    ARC objective. Keeping the *score* binary stops partial-credit reward hacking
    (you cannot inch the number up by getting backgrounds/sizes "closer").

    FEEDBACK gives the reflective optimizer a usable gradient despite the binary
    score: for each failed case it states how the prediction failed AND shows the
    correct output grid (plus the prediction beside it for shape-correct misses),
    rendered one row per line, so the optimizer can infer the rule it missed.

    NOTE: feedback intentionally reveals the gold output. That is the point of a
    reflective signal, but it means real generalization MUST be judged on the
    held-out TEST set, which is never read during the run (valset/test traces are
    deny-guarded so the coding agent cannot simply memorize them).
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
            notes.append(f"case {i}: no prediction returned. Correct output:{_grid_str(target)}")
            continue

        ph, pw = _dims(pred)
        th, tw = _dims(target)

        if ph < 0:
            case_scores.append(0.0)
            notes.append(
                f"case {i}: prediction is not a well-formed 2D grid. Correct output:{_grid_str(target)}"
            )
            continue
        if (ph, pw) != (th, tw):
            case_scores.append(0.0)
            notes.append(
                f"case {i}: wrong output dimensions — predicted {ph}x{pw}, expected {th}x{tw}. "
                f"Correct output:{_grid_str(target)}"
            )
            continue

        acc = _cell_accuracy(pred, target)
        if acc == 1.0:
            case_scores.append(1.0)
            exact += 1
        else:
            case_scores.append(0.0)
            wrong = round((1.0 - acc) * th * tw)
            notes.append(
                f"case {i}: correct shape ({th}x{tw}) but {wrong}/{th * tw} cells wrong. "
                f"Predicted:{_grid_str(pred)}\nCorrect output:{_grid_str(target)}"
            )

    score = sum(case_scores) / n  # each case_score is exactly 0.0 or 1.0

    summary = f"Solved {exact}/{n} test case(s) exactly."
    detail = " ".join(notes[:5])
    feedback = (summary + " " + detail).strip() if detail else summary

    return {"score": score, "feedback": feedback}
