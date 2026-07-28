from __future__ import annotations

from collections.abc import Iterable, Sequence

from .protocol import BudgetConfig


def expand_line_numbers(
    line_numbers: Iterable[int],
    *,
    line_count: int,
    window: int,
) -> set[int]:
    expanded: set[int] = set()
    for line_no in line_numbers:
        start = max(1, line_no - window)
        end = min(line_count, line_no + window)
        expanded.update(range(start, end + 1))
    return expanded


def select_line_numbers(
    scores: Sequence[float],
    budget: BudgetConfig,
    *,
    mandatory: Iterable[int] = (),
    expansion_seeds: Iterable[int] = (),
) -> tuple[int, ...]:
    """Select lines deterministically while respecting a hard line budget.

    Mandatory lines win over ranked lines. Expansion candidates are admitted by
    score order and proximity but never make the result exceed the target.
    """

    line_count = len(scores)
    target = budget.target_lines(line_count)
    if target >= line_count:
        return tuple(range(1, line_count + 1))
    if target == 0:
        return ()

    valid_mandatory = {line_no for line_no in mandatory if 1 <= line_no <= line_count}
    ranked = sorted(
        range(1, line_count + 1),
        key=lambda line_no: (-float(scores[line_no - 1]), line_no),
    )

    selected: set[int] = set(
        sorted(
            valid_mandatory,
            key=lambda line_no: (-float(scores[line_no - 1]), line_no),
        )[:target]
    )

    seed_lines = tuple(expansion_seeds)
    expanded = expand_line_numbers(
        seed_lines,
        line_count=line_count,
        window=budget.context_window,
    )
    expansion_ranked = (
        sorted(
            expanded - selected,
            key=lambda line_no: (
                -float(scores[line_no - 1]),
                min(abs(line_no - seed) for seed in seed_lines),
                line_no,
            ),
        )
        if expanded and seed_lines
        else []
    )

    for line_no in expansion_ranked + ranked:
        if len(selected) >= target:
            break
        selected.add(line_no)
    return tuple(sorted(selected))


def render_pruned_text(
    lines: Sequence[str],
    kept_line_numbers: Iterable[int],
    *,
    show_line_numbers: bool = True,
) -> str:
    """Render a skeleton with explicit omission markers."""

    kept = sorted({line_no for line_no in kept_line_numbers if 1 <= line_no <= len(lines)})
    if not kept:
        return f"... <{len(lines)} lines pruned> ..." if lines else ""

    output: list[str] = []
    previous = 0
    for line_no in kept:
        gap = line_no - previous - 1
        if gap:
            output.append(f"... <{gap} lines pruned> ...")
        line = lines[line_no - 1]
        output.append(f"{line_no:>6} | {line}" if show_line_numbers else line)
        previous = line_no
    tail = len(lines) - previous
    if tail:
        output.append(f"... <{tail} lines pruned> ...")
    return "\n".join(output)
