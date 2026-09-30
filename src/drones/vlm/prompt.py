"""The words a VLM pilot is given with each frame.

The model sees one image and has no memory, so the prompt carries everything else: the question,
the altitude and time it cannot see, and what one action does in centimetres and degrees. Those
sizes come from `drones.config`, the limits the actions are executed with, so the prompt cannot
drift from the flight envelope.

The format is shown as one complete, valid chunk rather than described. A small model counts to
16 more reliably from an example than from an instruction.
"""
import json

from drones import config
from drones.vlm.actions import CHUNK, STEP, schema_for

ROLE = ('You are the pilot of a small indoor drone. The image is the view from its forward '
        'camera right now.')
EXPLORE = 'There is no question. Your task: explore the space without flying into anything.'
DONE = '{"actions": [], "done": true, "answer": "your answer"}'
# Characters of a rejected reply and of its error shown on a retry. A reply can run to the token
# limit and pydantic reports every bad action, so both are cut.
REPLY_SHOWN = 1000
ERROR_SHOWN = 500


def example(space):
    """A complete valid chunk in action space `space`, as JSON: ahead, then a left turn."""
    schema_for(space)
    if space == 'discrete':
        actions = ['forward'] * 12 + ['turn_left'] * 4
    else:
        actions = ([{'forward': 1.0, 'yaw': 0.0, 'altitude': None}] * 12
                   + [{'forward': 0.0, 'yaw': 1.0, 'altitude': None}] * 4)
    return json.dumps({'actions': actions, 'done': False, 'answer': None})


def _actions(space):
    ahead = config.MAX_MANUAL_SPEED * STEP * 100    # cm per action at full scale
    turn = config.MAX_YAW_RATE * STEP               # degrees
    climb = config.MAX_CLIMB_SPEED * STEP * 100     # cm
    if space == 'discrete':
        return ['Each action is one of these words:',
                f'- "forward", "backward": move {ahead:.1f} cm.',
                f'- "turn_left", "turn_right": turn {turn:.1f} degrees.',
                f'- "rise", "descend": climb or sink {climb:.1f} cm.',
                '- "hover": stay still.']
    return ['Each action is {"forward": F, "yaw": Y, "altitude": H}:',
            f'- F is from -1 to 1. 1 moves {ahead:.1f} cm forward, -1 the same distance back.',
            f'- Y is from -1 to 1. 1 turns {turn:.1f} degrees left, -1 the same angle right.',
            f'- H is the altitude to fly to, from {config.MIN_ALTITUDE:g} to '
            f'{config.MAX_ALTITUDE:g} m, or null to stay level. One action climbs or sinks at '
            f'most {climb:.1f} cm.']


def build_prompt(space, question=None, choices=(), altitude=0.0, elapsed=0.0):
    """The prompt for one chunk: `question` (None to explore) with its multiple `choices`, the
    drone `altitude` m above the floor, `elapsed` s into the episode."""
    schema_for(space)
    lines = [ROLE, '']
    if question:
        lines.append(f'Your task: {question}')
        if choices:
            lines += ['The possible answers:', *choices]
    else:
        lines.append(EXPLORE)
    lines += ['',
              f'The drone is {altitude:.2f} m above the floor and has flown for {elapsed:.0f} s.',
              '',
              f'Choose its next {CHUNK} actions. They are played one after another over the next '
              'second, and then you are shown the new view.',
              *_actions(space),
              '',
              f'Reply with one JSON object and nothing else, holding exactly {CHUNK} actions. '
              'For example:',
              example(space),
              '',
              'When the task is complete, reply like this instead:',
              DONE]
    if question and choices:
        lines.append('Answer with the letter of your choice.')
    return '\n'.join(lines)


def retry_prompt(prompt, reply, error):
    """`prompt` again, after the `reply` that was rejected and the `error` that rejected it."""
    return (f'{prompt}\n\nYour last reply was rejected:\n{reply[:REPLY_SHOWN]}\n\n'
            f'The error: {error[:ERROR_SHOWN]}\n\nReply again with valid JSON only.')
