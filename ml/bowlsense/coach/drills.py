"""Drill library: 2-3 practical drills for every technique fault code."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Drill(BaseModel):
    """One drill."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    reps: str


DRILLS: dict[str, tuple[Drill, ...]] = {
    "KNEE_COLLAPSE": (
        Drill(name="Front-leg brace walk-throughs",
              description="Walk through your last two strides and land on a firm, nearly "
              "straight front leg; hold the braced position for two seconds after release.",
              reps="3 x 8"),
        Drill(name="Single-leg landings",
              description="Hop onto your front foot from a low box and stop dead with the knee "
              "soft but not collapsing; keep the hip over the foot.",
              reps="3 x 6 each leg"),
        Drill(name="Brace-and-bowl from a standing start",
              description="From a stationary delivery stride, bowl into a target focusing only "
              "on keeping the front knee firm through release.",
              reps="2 x 12 balls"),
    ),
    "OVERSTRIDE": (
        Drill(name="Stride markers",
              description="Place a marker about 80% of your height from the back-foot landing "
              "and land the front foot on or just before it.",
              reps="3 x 6 balls"),
        Drill(name="Three-step approach",
              description="Bowl off a three-step run so the gather stays compact and the "
              "delivery stride shortens naturally.",
              reps="2 x 12 balls"),
    ),
    "UNDERSTRIDE": (
        Drill(name="Long-stride bounding",
              description="Bound into the crease and reach for a marker about 75% of your "
              "height away, landing balanced.",
              reps="3 x 6"),
        Drill(name="Stride-length markers",
              description="Bowl with a marker placed at your target stride and aim to land the "
              "front foot level with it.",
              reps="3 x 6 balls"),
    ),
    "HEAD_FALL": (
        Drill(name="Eyes-level target bowling",
              description="Keep your eyes level and on the target until after release; a "
              "partner checks your head stays over the front foot.",
              reps="3 x 6 balls"),
        Drill(name="Wall-head drill",
              description="Stand in your front-foot-contact position beside a wall and bowl "
              "the arm through without letting the head touch the wall.",
              reps="2 x 10 slow reps"),
    ),
    "LATERAL_FLEXION": (
        Drill(name="Tall-at-release shadow bowling",
              description="Shadow bowl in front of a mirror, keeping the trunk tall and the "
              "shoulders level instead of bending away from the bowling arm.",
              reps="3 x 10"),
        Drill(name="Side-plank holds",
              description="Build trunk strength that resists side bend: side planks on each "
              "side.",
              reps="3 x 30 s each side"),
        Drill(name="Vertical-arm target bowling",
              description="Bowl with a high target (a cone on a stump) to encourage a tall, "
              "upright release without leaning over.",
              reps="2 x 12 balls"),
    ),
    "WEAK_FRONT_ARM": (
        Drill(name="High front-arm pull-downs",
              description="At front-foot contact reach the front arm high towards the target, "
              "then pull it down firmly to the hip as the bowling arm comes over.",
              reps="3 x 10 shadow reps"),
        Drill(name="Front-arm target drill",
              description="Point the front arm at a target above head height at front-foot "
              "contact on every ball.",
              reps="2 x 12 balls"),
    ),
    "RUNUP_DECEL": (
        Drill(name="Accelerating run-throughs",
              description="Run through the crease without bowling, building speed steadily all "
              "the way so the last strides are the quickest.",
              reps="5 runs"),
        Drill(name="Marked run-up rhythm",
              description="Mark your run-up start and count the last six strides out loud to "
              "keep the rhythm building into the crease.",
              reps="2 x 6 balls"),
    ),
    "LOW_RELEASE": (
        Drill(name="Tall-release target bowling",
              description="Release from as high as you comfortably can over a raised target "
              "(e.g. a cone on a stump), keeping the bowling arm close to the head.",
              reps="3 x 6 balls"),
        Drill(name="Front-leg block drill",
              description="A firmer front leg lifts the whole body at release: practise "
              "bracing the front leg from a standing start.",
              reps="2 x 10 balls"),
    ),
    "CHAIN_ORDER": (
        Drill(name="Hips-first medicine-ball throws",
              description="Throw a light medicine ball against a wall, leading with the hips, "
              "then trunk, then arms, to feel the sequence.",
              reps="3 x 8 each side"),
        Drill(name="Slow-motion sequencing",
              description="Bowl at half pace thinking 'hips, chest, arm' through the delivery "
              "stride.",
              reps="2 x 10 balls"),
    ),
    "SHORT_FOLLOW": (
        Drill(name="Follow-through to a marker",
              description="Let the body carry through after release and finish past a marker "
              "a few metres beyond the crease.",
              reps="3 x 6 balls"),
        Drill(name="Momentum walk-through",
              description="After each ball, take three deliberate steps down the pitch line "
              "before stopping (off the danger area).",
              reps="2 x 12 balls"),
    ),
}  # fmt: skip


def drills_for(code: str) -> tuple[Drill, ...]:
    """The drills for a fault code (empty if unknown)."""
    return DRILLS.get(code, ())
