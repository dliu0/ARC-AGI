def _extract(args, kwargs):
    """Pull (train_pairs, test_cases, predictions) out of either DSPy-style
    positional args (example, pred, trace) or the keyword form the evaluator
    passes."""
    train_pairs = None
    test_cases = None
    predictions = None

    if len(args) >= 2:
        example, pred = args[0], args[1]
        if hasattr(example, "train"):
            train_pairs = example.train
        elif isinstance(example, dict):
            train_pairs = example.get("train")
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

    # Keyword form the mounted evaluator uses: fn(output=..., example=Example(row)).
    # Pull train/test off the example object (attr- or dict-style). Without this,
    # test_cases stays empty and every row scores 0.0 ("No test cases available").
    if train_pairs is None and "example" in kwargs:
        example = kwargs["example"]
        if hasattr(example, "train"):
            train_pairs = example.train
        elif isinstance(example, dict):
            train_pairs = example.get("train")
        if hasattr(example, "test"):
            test_cases = example.test
        elif isinstance(example, dict):
            test_cases = example.get("test")

    if train_pairs is None:
        train_pairs = kwargs.get("train")
    if test_cases is None:
        test_cases = kwargs.get("test")
    if predictions is None:
        predictions = (
            kwargs.get("predictions")
            or kwargs.get("pred")
            or kwargs.get("output")
        )
    return train_pairs or [], test_cases or [], predictions or []


def _norm_cell(x):
    """Normalize a cell so a string "3" and int 3 compare equal.

    ARC cell values are the digits 0-9. Models sometimes emit them as strings;
    treating "3" and 3 as different would zero out an otherwise-correct grid,
    which is a scoring artifact, not a real miss. Anything that is not an
    int-like value is returned unchanged so genuinely wrong cells still differ.
    """
    if isinstance(x, bool):
        return x
    try:
        return int(x)
    except (TypeError, ValueError):
        return x


def _cells_equal(a, b):
    return _norm_cell(a) == _norm_cell(b)


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
    """Fraction of matching cells (type-normalized); assumes equal dimensions."""
    total = 0
    correct = 0
    for prow, trow in zip(pred, target):
        for pcell, tcell in zip(prow, trow):
            total += 1
            if _cells_equal(pcell, tcell):
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


def _values(grid):
    """Sorted set of distinct cell values in a 2D grid (normalized)."""
    vals = set()
    if isinstance(grid, list):
        for row in grid:
            if isinstance(row, list):
                for c in row:
                    vals.add(_norm_cell(c))
    return sorted(vals, key=lambda v: (str(type(v)), v))


def _cell_diff(pred, target, cap=40):
    """List of mismatched cells as 'row,col: got G want W', capped.

    Assumes equal dimensions (callers only invoke this on shape-correct misses).
    Returns (lines, truncated).
    """
    lines = []
    for r, (prow, trow) in enumerate(zip(pred, target)):
        for c, (p, t) in enumerate(zip(prow, trow)):
            if not _cells_equal(p, t):
                lines.append(f"r{r}c{c}: got {p} want {t}")
                if len(lines) >= cap:
                    return lines, True
    return lines, False


def _demos_str(train_pairs, cap_pairs=4):
    """Render the demonstration (input -> output) pairs the rule is inferred from."""
    if not train_pairs:
        return ""
    chunks = []
    for j, pair in enumerate(train_pairs[:cap_pairs]):
        if isinstance(pair, dict):
            inp, out = pair.get("input"), pair.get("output")
        else:
            inp, out = None, None
        chunks.append(
            f"demo {j} input:{_grid_str(inp)}\ndemo {j} output:{_grid_str(out)}"
        )
    extra = "" if len(train_pairs) <= cap_pairs else f"\n(+{len(train_pairs) - cap_pairs} more demonstration pairs)"
    header = (
        "Demonstration pairs (infer the single transformation rule that maps "
        "every input to its output, then apply it to the test input):\n"
    )
    return header + "\n".join(chunks) + extra


def arc_grid_accuracy(*args, **kwargs):
    """Strict ARC grid metric with answer-revealing, how-to-solve feedback.

    Returns ``{"score": float, "feedback": str}``.

    SCORE is strictly 0.0 or 1.0 per test case (exact match required) — the real
    ARC objective. Keeping the *score* binary stops partial-credit reward hacking
    (you cannot inch the number up by getting backgrounds/sizes "closer"). The
    only leniency is type normalization: a string "3" counts as the int 3, since
    that is the same ARC color, not a closer-but-wrong answer.

    FEEDBACK gives the reflective optimizer a rich, usable gradient despite the
    binary score. For the task it shows the demonstration pairs (the rule to
    infer), and for each failed case it shows the test INPUT, the prediction,
    the correct OUTPUT, an exact cell-by-cell diff for shape-correct misses, and
    the colors involved — everything needed to reason about the missed rule.

    NOTE: feedback intentionally reveals the gold output. That is the point of a
    reflective signal, but it means real generalization MUST be judged on the
    held-out TEST set, which is never read during the run (valset/test traces are
    deny-guarded so the coding agent cannot simply memorize them).
    """
    train_pairs, test_cases, predictions = _extract(args, kwargs)

    if not test_cases:
        return {"score": 0.0, "feedback": "No test cases were available to score."}

    n = len(test_cases)
    case_scores = []
    notes = []
    exact = 0

    for i in range(n):
        case = test_cases[i] if isinstance(test_cases[i], dict) else {}
        target = case.get("output", [])
        test_input = case.get("input", [])
        pred = predictions[i] if i < len(predictions) else None

        in_str = f"Test input:{_grid_str(test_input)}\n" if test_input else ""

        if pred is None:
            case_scores.append(0.0)
            notes.append(
                f"case {i}: NO prediction returned (the program produced no grid for "
                f"this test). {in_str}Correct output:{_grid_str(target)}"
            )
            continue

        ph, pw = _dims(pred)
        th, tw = _dims(target)

        if ph < 0:
            case_scores.append(0.0)
            notes.append(
                f"case {i}: prediction is not a well-formed 2D grid (got {pred!r:.200}). "
                f"{in_str}Correct output:{_grid_str(target)}"
            )
            continue
        if (ph, pw) != (th, tw):
            case_scores.append(0.0)
            notes.append(
                f"case {i}: WRONG output dimensions — predicted {ph}x{pw}, expected "
                f"{th}x{tw}. First decide the output size from the rule. {in_str}"
                f"Your prediction:{_grid_str(pred)}\nCorrect output:{_grid_str(target)}"
            )
            continue

        acc = _cell_accuracy(pred, target)
        if acc == 1.0:
            case_scores.append(1.0)
            exact += 1
        else:
            case_scores.append(0.0)
            wrong = round((1.0 - acc) * th * tw)
            diff_lines, truncated = _cell_diff(pred, target)
            diff_str = "; ".join(diff_lines) + (" ..." if truncated else "")
            pv, tv = _values(pred), _values(target)
            color_note = (
                f" Colors used — predicted {pv}, expected {tv}." if pv != tv else ""
            )
            notes.append(
                f"case {i}: correct shape ({th}x{tw}) but {wrong}/{th * tw} cells wrong."
                f"{color_note} Mismatched cells [{diff_str}]. {in_str}"
                f"Your prediction:{_grid_str(pred)}\nCorrect output:{_grid_str(target)}"
            )

    score = sum(case_scores) / n  # each case_score is exactly 0.0 or 1.0

    summary = f"Solved {exact}/{n} test case(s) exactly."
    demos = _demos_str(train_pairs)
    detail = "\n\n".join(notes[:3])
    parts = [summary]
    if detail:
        parts.append(detail)
    if demos:
        parts.append(demos)
    feedback = "\n\n".join(parts).strip()

    return {"score": score, "feedback": feedback}
