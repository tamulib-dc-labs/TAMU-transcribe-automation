# Output files

Two files per interview, named after the recording:

```
data/oral_output/json/02_00113_a_01.json     the data
data/oral_output/vtts/02_00113_a_01.vtt      the subtitles
```

The format is **the same as v2.0**, so the reviewer app works with either
version. v3.0 adds a few fields; nothing was removed or renamed.

---

## The subtitle file

```
WEBVTT

00:00:00.420 --> 00:00:03.440
<v S01>So tell me about the summer of nineteen
sixty two.

00:00:04.180 --> 00:00:09.310
<v S02>Well, we were still out on the Brazos
place then. My father worked the land and

00:00:09.310 --> 00:00:11.680
<v S02>my mother taught school in Navasota.
```

`<v S01>` is standard WebVTT speaker markup. Players that understand it show
the speaker; players that do not simply ignore it.

Lines wrap at 42 characters, two per cue, and **never mix two speakers in one
cue**.

Cue start and end times come from the **words**, not from the speaker turn — so
every break falls on actual speech rather than in a silence.

---

## The JSON file

### Overall shape

```json
{
  "segments": [ ... ],
  "language": "en",
  "word_score_buckets": { "Good": 0.95, "Neutral": 0.94, "Bad": 0.91 },
  "alignment_stats": { ... },
  "speakers": ["S01", "S02"],
  "asr_model": "nvidia/parakeet-tdt-0.6b-v3"
}
```

### A segment

One segment is one uninterrupted stretch by one speaker.

```json
{
  "start": 4.18,
  "end": 11.68,
  "text": "Well, we were still out on the Brazos place then. My father
           worked the land and my mother taught school in Navasota.",
  "speaker": "S02",
  "words": [ ... ]
}
```

### A word

```json
{
  "word": "Brazos",
  "start": 5.87,
  "end": 6.44,
  "score": 0.58,
  "speaker": "S02",
  "source": "measured"
}
```

| Field | Meaning |
|---|---|
| `word` | The word, with its punctuation |
| `start` / `end` | Seconds from the beginning of the recording |
| `score` | How confident the model was, from 0 to 1 |
| `speaker` | Who said it. `null` if diarization was off or failed |
| `source` | `measured` = the model timed it from the audio. `interpolated` = estimated, because the model gave no timing |

In the example above, `Brazos` scores **0.58** while its neighbours are above
0.90. That is a local place name the model was unsure of — exactly the kind of
thing a reviewer should check.

---

## What `score` means, and what it does not

`score` says **how strongly the audio supports that word**. It does *not* say
the word is correct.

- A **low** score means "check this one".
- A **high** score does **not** guarantee correctness — a confidently misheard
  word scores high.
- Short function words (`a`, `of`, `the`) naturally score lower than content
  words. Compare like with like.

---

## `word_score_buckets`

```json
"word_score_buckets": { "Good": 0.95, "Neutral": 0.94, "Bad": 0.91 }
```

These are the 75th, 50th and 25th percentiles **of that file's own scores**.

Because they are percentiles rather than fixed numbers, they adjust themselves
per recording — a clean interview and a noisy one each get sensible thresholds.
That is also why changing the model did not break the reviewer app.

---

## `alignment_stats`

```json
{
  "total_segments": 3,
  "successful_alignments": 3,
  "failed_alignments": 0,
  "success_rate": 100.0,
  "total_words": 39,
  "words_with_measured_timings": 39
}
```

The name is kept from v2.0 for compatibility. The number to watch is
**`words_with_measured_timings`** against `total_words` — if a lot of words were
interpolated rather than measured, look at the audio quality.

---

## New in v3.0

All additions. Nothing was removed, so old consumers keep working.

| Field | Where | What |
|---|---|---|
| `speaker` | segment and word | Who was talking |
| `source` | word | `measured` or `interpolated` |
| `speakers` | top level | Every speaker found in the file |
| `asr_model` | top level | Which model produced this — useful when comparing runs |

---

## Reading it in Python

```python
import json

with open("data/oral_output/json/02_00113_a_01.json") as f:
    data = json.load(f)

# Every word below a confidence threshold
for segment in data["segments"]:
    for word in segment["words"]:
        if word["score"] is not None and word["score"] < 0.6:
            print(f"{word['start']:7.2f}s  {word['speaker']}  {word['word']}  ({word['score']})")

# How much each person talked
from collections import defaultdict
talk = defaultdict(float)
for segment in data["segments"]:
    talk[segment["speaker"]] += segment["end"] - segment["start"]
for speaker, seconds in talk.items():
    print(f"{speaker}: {seconds / 60:.1f} minutes")
```

---

## Speaker labels

Speakers are `S01`, `S02`, and so on, in the order they first speak. The model
does not know anyone's name — `S01` is usually the interviewer simply because
interviewers usually speak first, but check before assuming.

Labels are consistent **within one file** and mean nothing across files: `S01`
in one interview is not the same person as `S01` in another.

The model handles **up to 4 speakers**. Beyond that, accuracy falls off sharply.
