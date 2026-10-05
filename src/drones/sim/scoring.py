"""Score benchmark runs: drones-benchmark score runs/benchmarks/*

Reads what drones.sim.benchmark wrote and computes FAST-EQA's metrics per benchmark (see
drones.sim.metrics for where each comes from), then writes scores.json (a row per question)
and report.md beside results.jsonl.

Open answers (EXPRESS-Bench, A-EQA) are marked 1 to 5 by a judge with OpenEQA's own prompt,
word for word. The papers used GPT-4 or GPT-4o-mini; here it is Claude through the claude CLI,
the same subscription that flew. Marks are cached in judged.jsonl by question, answer and judge
model, so scoring again judges only what is new, and another --judge-model judges everything
again. A judge that fails stops scoring as a run stops, with every mark so far kept.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from drones.sim import metrics
from drones.sim.benchmark import Stopped, finished

JUDGE_MODEL = 'sonnet'
JUDGE_SYSTEM = 'You mark answers. Follow the instructions in the message.'
MULTIPLE_CHOICE = ('hm-eqa', 'mt-hm3d')
OPEN = ('express-bench', 'a-eqa')
# OpenEQA's prompts/mmbench.txt at commit cfa3fce, verbatim.
JUDGE_PROMPT = '''You are an AI assistant who will help me to evaluate the response given the question and the correct answer.
To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
5 means that the response perfectly matches the answer.
1 means that the response is completely different from the answer.

Example 1:
Question: Is it overcast?
Answer: no
Response: yes
Your mark: 1

Example 2:
Question: Who is standing at the table?
Answer: woman
Response: Jessica
Your mark: 3

Example 3:
Question: Are there drapes to the right of the bed?
Answer: yes
Response: yes
Your mark: 5

Your Turn:
Question: {question}
Answer: {answer}
Response: {prediction}'''   # noqa: E501
DEVIATIONS = [
    'The judge is Claude through the claude CLI, not GPT-4 (OpenEQA) or GPT-4o-mini '
    '(EXPRESS-Bench).',
    "EXPRESS-Bench's grounding term is not computed: its C* is reported, and E_path takes the "
    'grounding as 1.',
    'Distances are straight lines in the scan, not geodesics on a navmesh: there is no '
    'habitat-sim here, and the drone flies through walls.',
    "A step is one model call, which flies at most `reach` (0.4 m by default); Explore-EQA's "
    'step moves up to 3 m.',
    "The floor area behind the EQA budget is the box of the scan's vertices 0.1 to 2.0 m above "
    "the start's floor, not habitat's navmesh bounds.",
    'A-EQA has no reference path here, so it has no E_path.',
    'nDTW is computed on both paths resampled every 0.5 m, so that it does not depend on how '
    'densely the flight is sampled.',
]


class Judge:
    """Marks answers 1 to 5 through `backend` (generate(prompt, image) -> str), as `model`,
    caching each mark in `cache_path`."""

    def __init__(self, backend, model, cache_path):
        self.backend, self.model, self.cache_path = backend, model, Path(cache_path)
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.cache[(row['number'], row['answer'], row['model'])] = row['mark']

    def mark(self, number, question, truth, answer):
        """The mark for `answer`; 1 without asking for no answer at all."""
        if answer is None or not str(answer).strip():
            return 1
        key = (number, answer, self.model)
        if key in self.cache:
            return self.cache[key]
        prompt = JUDGE_PROMPT.format(question=question, answer=truth, prediction=answer)
        error = None
        for _ in range(2):      # one retry, as the pilot gets
            try:
                mark = metrics.parse_mark(self.backend.generate(prompt, None))
                break
            except ValueError as exc:
                error = exc
        else:
            raise Stopped(f'the judge failed on question {number}: {error}')
        self.cache[key] = mark
        with open(self.cache_path, 'a') as f:
            f.write(json.dumps({'number': number, 'answer': answer, 'model': self.model,
                                'mark': mark}) + '\n')
        return mark


def claude_judge(folder, model=JUDGE_MODEL):
    """A Judge asking Claude `model` through the claude CLI, caching in `folder`."""
    from drones.vlm.backend import ClaudeCodeBackend

    return Judge(ClaudeCodeBackend(model, system=JUDGE_SYSTEM), model,
                 Path(folder) / 'judged.jsonl')


def _category(benchmark, category):
    """IndoorUAV's categories begin with the trajectory's own name; group by split and
    difficulty."""
    if benchmark == 'indoor-uav':
        return category.split(', ', 1)[1] if ', ' in category else category
    return category


def _row(folder, benchmark, result, judge):
    row = {key: result[key] for key in ('number', 'stop', 'decisions', 'budget', 'answer',
                                        'truth')}
    row['category'] = _category(benchmark, result.get('category') or '')
    if benchmark in MULTIPLE_CHOICE:
        truth = metrics.choice_letter(result['truth'], result['choices'])
        picked = metrics.choice_letter(result['answer'], result['choices'])
        row['correct'] = truth is not None and picked == truth
    if benchmark in OPEN:
        try:
            row['mark'] = judge.mark(result['number'], result['question'], result['truth'],
                                     result['answer'])
        except Stopped:
            raise
        except Exception as exc:
            raise Stopped(f'the judge failed on question {result["number"]}: {exc}') from exc
    if result.get('goal') is not None:
        row['distance'] = float(np.linalg.norm(np.subtract(result['final_pos'],
                                                           result['goal'])))
    if benchmark == 'indoor-uav':
        path = Path(folder) / 'episodes' / str(result['number']) / 'trajectory.npz'
        if not path.exists():
            raise Stopped(f'question {result["number"]}: no episodes/{result["number"]}/'
                          f'trajectory.npz in {folder}')
        trajectory = np.load(path)
        row.update(metrics.navigation(trajectory['pos'], np.asarray(result['goal'], float)))
        row['ndtw'] = metrics.ndtw(trajectory['reference'], trajectory['pos'])
    return row


def _metrics(benchmark, rows, results):
    """The benchmark's metrics over `rows` (from _row) and their `results` lines."""
    m = {}
    if benchmark in MULTIPLE_CHOICE:
        m['SR'] = 100 * float(np.mean([r['correct'] for r in rows]))
        m['Normalized steps'] = float(np.mean([r['decisions'] / r['budget'] for r in rows]))
    if benchmark == 'express-bench':
        marks = [r['mark'] for r in rows]
        m['LLM-Score C*'] = metrics.llm_score(marks)
        m['E_path'] = metrics.e_path(marks, [x['reference_length'] or 0.0 for x in results],
                                     [x['path_length'] for x in results])
        near = [r['distance'] for r in rows if np.isfinite(r.get('distance', math.nan))]
        m['d_T (m)'] = float(np.mean(near)) if near else math.nan
    if benchmark == 'a-eqa':
        m['LLM-Match'] = metrics.llm_match([r['mark'] for r in rows])
    if benchmark == 'indoor-uav':
        m['SR'] = 100 * float(np.mean([r['success'] for r in rows]))
        m['OSR'] = 100 * float(np.mean([r['oracle'] for r in rows]))
        m['NE (m)'] = float(np.mean([r['ne'] for r in rows]))
        m['nDTW'] = 100 * float(np.mean([r['ndtw'] for r in rows]))
    m['Out of budget %'] = 100 * float(np.mean([r['stop'] == 'budget' for r in rows]))
    calls = sum(x['model_calls'] for x in results)
    m['Seconds per call'] = sum(x['model_seconds'] for x in results) / calls if calls else math.nan
    stats = [x['stats'] for x in results if x.get('stats')]
    total = sum(sum(s.values()) for s in stats)
    if total:
        m['Valid at once %'] = 100 * sum(s['first'] for s in stats) / total
        m['Valid after retry %'] = 100 * sum(s['retry'] for s in stats) / total
        m['Failed %'] = 100 * sum(s['failed'] for s in stats) / total
    return m


def score(folder, judge):
    """Score the run in `folder` with `judge`; writes scores.json and report.md there and
    returns the summary."""
    folder = Path(folder)
    benchmark = json.loads((folder / 'config.json').read_text())['benchmark']
    results = sorted(finished(folder / 'results.jsonl').values(), key=lambda r: r['number'])
    rows = [_row(folder, benchmark, result, judge) for result in results]
    by_category = defaultdict(list)
    for row, result in zip(rows, results, strict=True):
        by_category[row['category']].append((row, result))
    summary = {
        'benchmark': benchmark, 'folder': str(folder), 'questions': len(rows),
        'metrics': _metrics(benchmark, rows, results) if rows else {},
        'categories': {c: _metrics(benchmark, [r for r, _ in pairs], [x for _, x in pairs])
                       for c, pairs in sorted(by_category.items())},
    }
    (folder / 'scores.json').write_text(json.dumps(rows, indent=1))
    (folder / 'report.md').write_text(report([summary]))
    return summary


def _table(names, rows):
    head = '| ' + ' | '.join(names) + ' |\n|' + ' --- |' * len(names) + '\n'
    return head + ''.join('| ' + ' | '.join(row) + ' |\n' for row in rows)


def _cell(value):
    if isinstance(value, float):
        return '-' if math.isnan(value) else f'{value:.2f}'
    return str(value)


def report(summaries):
    """Markdown for one run or several: a line each, then each run by category, then the
    deviations from the papers."""
    names = []
    for s in summaries:
        names += [n for n in s['metrics'] if n not in names]
    lines = ['# Benchmark results', '',
             _table(['Run', 'Benchmark', 'Questions', *names],
                    [[Path(s['folder']).name, s['benchmark'], str(s['questions']),
                      *[_cell(s['metrics'].get(n, math.nan)) for n in names]]
                     for s in summaries])]
    for s in summaries:
        own = list(s['metrics'])
        lines += [f'## {Path(s["folder"]).name} by category', '',
                  _table(['Category', *own],
                         [[c, *[_cell(m.get(n, math.nan)) for n in own]]
                          for c, m in s['categories'].items()])]
    lines += ['## Deviations from the papers', '', *[f'- {d}' for d in DEVIATIONS], '']
    return '\n'.join(lines)
