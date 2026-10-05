"""The words a VLM pilot is given with each frame.

The model sees one image and has no memory, so the prompt carries everything else: the question,
the altitude and time it cannot see, and the action space, which is what each action does to the
drone and to the picture, in centimetres and degrees, and what a whole chunk of it adds up to.
Those sizes come from `drones.config`, the limits the actions are executed with, so the prompt
cannot drift from the flight envelope.

The reply format is a template with a slot for each action, never a finished chunk. The first
prompt showed one complete, valid example, and Gemma 3n E2B sent that example back for every
frame (with its turn reversed), whatever the image: it read the example as the answer. Nothing in
the prompt may therefore parse as a reply, which `tests/test_vlm_prompt.py` holds it to. The
slots are numbered so that a small model still has sixteen things to fill in, not a count to keep.
"""
from drones import config
from drones.vlm.actions import CHUNK, MOVES, STEP, schema_for

ROLE = ('You are the pilot of a small indoor drone. The image is the view from its forward '
        'camera right now.')
EXPLORE = 'There is no question. Your task: explore the space without flying into anything.'
# Characters of a rejected reply and of its error shown on a retry. A reply can run to the token
# limit and pydantic reports every bad action, so both are cut.
REPLY_SHOWN = 1000
ERROR_SHOWN = 500


def template(slot, done=False, size=CHUNK):
    """The reply format, with `slot`1 .. `slot``size` where the actions go. Not valid JSON. A
    done reply has the same slots: one shape is easier to keep to, and a model whose decoding is
    held to the schema can write no other (see actions.json_schema)."""
    slots = ', '.join(f'{slot}{i}' for i in range(1, size + 1))
    ending = '"done": true, "answer": ANSWER' if done else '"done": false, "answer": null'
    return '{"actions": [' + slots + '], ' + ending + '}'


def _action_space(space, size):
    """What the actions of `space` are and do, sized from the flight limits, and what `size`
    of them add up to."""
    ahead = config.MAX_MANUAL_SPEED * STEP * 100    # cm per action at full scale
    turn = config.MAX_YAW_RATE * STEP               # degrees
    climb = config.MAX_CLIMB_SPEED * STEP * 100     # cm
    far, around = ahead * size, turn * size         # a whole chunk of one action
    seconds = size * STEP
    over = 'second' if seconds == 1 else f'{seconds:g} seconds'
    each = f'1/{round(1 / STEP)} of a second'
    chunk = (f'You give {size} at a time. They are flown in order over the next {over}, and '
             'then you are shown the new view.')
    if space == 'discrete':
        return [f'ACTION SPACE. You fly with {len(MOVES)} moves. Each lasts {each}:',
                f'- "forward": fly {ahead:.1f} cm towards the centre of the image.',
                f'- "backward": fly {ahead:.1f} cm away from it.',
                f'- "turn_left": rotate {turn:.1f} degrees to the left. What is on the left of '
                'the image comes towards its centre.',
                f'- "turn_right": rotate {turn:.1f} degrees to the right. What is on the right '
                'of the image comes towards its centre.',
                f'- "rise": climb {climb:.1f} cm.',
                f'- "descend": sink {climb:.1f} cm.',
                '- "hover": stay still.',
                f'{chunk} Repeat a move to do more of it: {size} times "forward" is '
                f'{far:.0f} cm ahead, {size} times "turn_left" is {around:.0f} degrees to the '
                'left.']
    return ['ACTION SPACE. You fly with actions of the form {"forward": F, "yaw": Y, '
            f'"altitude": H}}. Each lasts {each}:',
            f'- F is a number from -1 to 1, the speed along the view. 1 flies {ahead:.1f} cm '
            'towards the centre of the image, -1 the same distance away from it, 0 stays in '
            'place, 0.5 is half of 1.',
            f'- Y is a number from -1 to 1, the turn. 1 rotates {turn:.1f} degrees to the left, '
            'so what is on the left of the image comes towards its centre; -1 the same angle to '
            'the right; 0 keeps the heading.',
            f'- H is the altitude to fly to in metres, from {config.MIN_ALTITUDE:g} to '
            f'{config.MAX_ALTITUDE:g} m, or null to stay at this height. One action climbs or '
            f'sinks at most {climb:.1f} cm.',
            'F and Y act together: F 1 with Y 0.5 flies a curve to the left.',
            f'{chunk} Repeat an action to do more of it: {size} times F 1 is {far:.0f} cm '
            f'ahead, {size} times Y 1 is {around:.0f} degrees to the left.']


def _how_to_choose(question):
    if question:
        seek = ['- If what the task names is in view, turn until it is in the centre of the '
                'image, then fly forward to it.',
                '- If it is not in view, turn to look around.']
    else:
        seek = ['- If the way ahead is open, fly forward.',
                '- Turn now and then, to see the parts of the space you have not seen.']
    return ['HOW TO CHOOSE. Look at the image before you choose. Your actions must fit this '
            'image, not any other.',
            *seek,
            '- Do not fly forward into a wall or an obstacle that fills the centre of the '
            'image. Turn towards the open side first.']


def _reply(space, question, choices, size):
    if space == 'discrete':
        slot, what = 'm', f'one of the {len(MOVES)} moves, in double quotes'
    else:
        slot, what = 'a', 'an action {"forward": F, "yaw": Y, "altitude": H} with your numbers'
    lines = ['REPLY. One JSON object and nothing else:',
             template(slot, size=size),
             f'with each of {slot}1 to {slot}{size} replaced by {what}. There are exactly '
             f'{size}.',
             'When the task is complete, reply like this instead:',
             template(slot, done=True, size=size),
             'with ANSWER replaced by your answer in double quotes. Those actions are not flown, '
             f'so any {size} will do.']
    if question and choices:
        lines.append('Your answer is the letter of your choice.')
    return lines


def build_prompt(space, question=None, choices=(), altitude=0.0, elapsed=0.0, size=CHUNK):
    """The prompt for one chunk of `size` actions: `question` (None to explore) with its
    multiple `choices`, the drone `altitude` m above the floor, `elapsed` s into the episode."""
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
              *_action_space(space, size),
              '',
              *_how_to_choose(question),
              '',
              *_reply(space, question, choices, size)]
    return '\n'.join(lines)


# The last section of the prompt when a benchmark's budget has run out: the model must answer
# now. The reply format is the template shown above it, so this holds no JSON of its own.
FINAL = ('THIS IS YOUR LAST LOOK. The flight is over: answer now, from this image and the task. '
         'Reply in the second form above, with "done" set to true and your answer.')


def final_prompt(space, question=None, choices=(), altitude=0.0, elapsed=0.0, size=CHUNK):
    """`build_prompt`'s prompt with a last section that asks for the answer now."""
    return f'{build_prompt(space, question, choices, altitude, elapsed, size)}\n\n{FINAL}'


def retry_prompt(prompt, reply, error):
    """`prompt` again, after the `reply` that was rejected and the `error` that rejected it."""
    return (f'{prompt}\n\nYour last reply was rejected:\n{reply[:REPLY_SHOWN]}\n\n'
            f'The error: {error[:ERROR_SHOWN]}\n\nReply again with valid JSON only.')
