# Live webcam tests (Thurushi, Step 6a, 2026-11-11)

`predict_live_ctc.py` on a MacBook webcam, signed by Thurushi. Small, informal tests (one person, a few tries per
word), so the numbers are rough; they show whether the live app behaves like the offline evaluation, and where it fails.
Scored with `metrics.py` against the words that were signed.

## Test 1: word list (model `models/step6_check_thurushi.pt`, never trained on Thurushi; greedy decoding)

About 25 words signed one at a time, several passes; scored on the third full pass (first try of each word).

| Group | CER | Words fully right | Letter recall |
|---|---|---|---|
| 15 words from our list (HI ... QUIZ) | 29% | 27% | 73% |
| 4 words not on our list (SUN, BLUE, ANNA, THURUSHI), best try | 32% | 0% | 68% |
| *Offline, same model, Thurushi's recorded clips* | *25%* | *34%* | *76%* |

Live is close to offline, so the live path (word cutting, resampling, features) matches training
(see also `python live_ctc.py --replay step6_check`: 484/495 recorded clips replay to exactly the offline prediction).

What went wrong, most important first:
- **Double letters merged:** HELLO→HELO, SCHOOL→SCHOL/CHOL, JAZZ→JAZ, PIZZA→PIZA, ANNA→ANN; sometimes an extra repeat (MMY, BIRRTHY).
- **Words not on the list never fully right:** THURUSHI→THRSHI, BLUE→BLRE, SUN→NRN, ANNA→ANN.
- **Long words hard:** BIRTHDAY right about 1 in 7 tries, WEEKEND never.
- **Repeated confusions:** K→X (WEEKEND→WEEXEND every time), Q→P (QUIZ never right), F→B (FAMILY→BAMILY).
- **Fast signing:** slow HELLO right, fast HELLO→HIZL; fast SCHOOL was split into two words (hand probably lost ~1 s).
- **Stray one-letter words** (G, P, Y) from a hand passing into view: the 5-frame minimum is too low.

What worked: short words (HI, NO, MY); J and Z (JUMP, ZOO); S (YES, SORRY, a lone S every time); the 1 s wait felt fine.

## Test 2: a sentence, both models (beam search + confidence overlay)

Signed: **HI MY NAME IS THURUSHI YOU LIKE JAZZ** (THURUSHI, YOU, LIKE and IS are not on our word list).

| Signed | Final model (`step6_final.pt`, trained on all 3 of us) | Held-out model (`step6_check_thurushi.pt`, never saw Thurushi) |
|---|---|---|
| HI | HI 100% | HI 97% |
| MY | MY 75% | MY 89% |
| NAME | NAME 52% | NME 88% |
| IS | J 57% | IS 71% |
| THURUSHI | THRSHI 64% | THTI 41% |
| YOU | OR 38% | OR 96% |
| LIKE | LKE 99% | LIE 40% |
| JAZZ | JAZ 95% | JZ 47% |
| **CER: list words / not-on-list words** | **8% / 41%** | **25% / 47%** |

- Same weaknesses in a natural sentence: double letters (JAZZ), words not on the list, **I vs J** (IS→J; J is an I-shape with a twist).
- Training on Thurushi did not clearly help live (each model got one word the other missed); 8 words is too few to say.
- **Confidence is useful but can be confidently wrong**, mostly on words not on the list: LKE 99% and OR 96% were wrong,
  while low scores (THTI 41%, LIE 40%) did flag errors.
- **The practice-word list (w key) fixes list words and invents wrong ones for everything else:** NME→NAME, JZ→JAZZ, but
  THURUSHI→HI, LIKE→KEY/LOVE, IS→YES, YOU→NO/JOB. This is why it is an optional extra line, never the main output.
