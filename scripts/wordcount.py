"""wordcount.py [SECTION...]: words of prose and figure captions in sec/training.typ's sections (default 9.4 and
9.5), excluding Typst code, comments, table cells and figure labels; the measure used for the second pass's cut."""
import re
import sys

text = open("sec/training.typ").read()
heads = {"9.4": "== Supervised Fine-tuning", "9.5": "== Reinforcement learning with Miles"}
ends = {"9.4": '#import "fig-rl-learning.typ"', "9.5": None}
for name in sys.argv[1:] or ["9.4", "9.5"]:
    seg = text[text.index(heads[name]):]
    if ends[name]:
        seg = seg[:seg.index(ends[name])]
    words = 0
    for line in seg.split("\n")[1:]:
        s = line.strip()
        if s.startswith("caption:"):
            s = s[len("caption:"):]
        elif not s or s[0] in "/#=[)]" or re.match(r"^(table|placement|columns|align|fig-|rl-learning)", s):
            continue
        words += len(s.split())
    print(name, words)
