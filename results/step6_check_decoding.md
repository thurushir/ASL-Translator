# Decoding comparison: greedy vs beam search vs word list (step6_check)

495 clips. Every clip is decoded by the model that never trained on its signer (`models/step6_check_<signer>.pt`), so all rows are held-out-signer scores. Same model outputs; only the decoder differs.

- **Beam search**: width 10, fixed before any results.
- **Word list**: picks the list word the clip most likely spells. It can only output our 55 words, all of which the model trained on, so it **flatters** the model and can never spell a new word. Report it next to greedy/beam, never instead of them.

| Decoder | Signer | CER | Word acc | Letter recall |
|---|---|---|---|---|
| greedy | anna | 27.7% | 20.6% | 85.8% |
| greedy | saaya | 9.4% | 66.1% | 92.2% |
| greedy | thurushi | 25.0% | 33.9% | 75.7% |
| greedy | all | 20.7% | 40.2% | 84.6% |
| beam (width 10) | anna | 28.5% | 19.4% | 86.2% |
| beam (width 10) | saaya | 9.1% | 66.7% | 93.6% |
| beam (width 10) | thurushi | 24.7% | 33.3% | 76.3% |
| beam (width 10) | all | 20.8% | 39.8% | 85.4% |
| word list (55 words) | anna | 1.4% | 97.6% | 99.0% |
| word list (55 words) | saaya | 0.6% | 99.4% | 99.6% |
| word list (55 words) | thurushi | 7.0% | 90.3% | 93.0% |
| word list (55 words) | all | 3.0% | 95.8% | 97.2% |

## Clips where beam search changed the greedy answer (43)

| Signer | Word | Greedy | Beam |
|---|---|---|---|
| saaya | APPLE | ALE | AHLE |
| saaya | APPLE | ALE | APLE |
| saaya | APPLE | APLEE | APPLEE |
| anna | BOX | BOOX | BBOOX |
| thurushi | COFFEE | COFE | COFFE |
| thurushi | DOG | DEG | DDEG |
| saaya | EGG | EG | EGG |
| saaya | EGG | EG | EGG |
| anna | EXTRA | FXTRA | FEXTRA |
| saaya | EXTRA | EXARA | EXATRA |
| saaya | GOOD | GOD | GOOD |
| anna | HI | HI | HIZ |
| anna | JAZZ | FIAZZZZZIL | FIAZZZZZII |
| anna | JOB | FIOB | FJOB |
| anna | JUICE | FIUICE | FIUICEE |
| anna | LETTER | FLETTER | FLLETTER |
| anna | LETTER | FLEER | FLETER |
| saaya | LETTER | LETTERK | LEETTERK |
| thurushi | LETTER | LENR | LENER |
| thurushi | LETTER | LETR | LETRR |
| thurushi | LOVE | LOVE | LORVE |
| anna | MILK | FILK | FMILK |
| saaya | NAME | NAME | TNAME |
| saaya | PIZZA | IZZA | HIZZA |
| saaya | PIZZA | PIZZZA | PIZZZZA |
| saaya | PLEASE | PLEASE | PPLEASE |
| saaya | PLEASE | PLEASE | PPLEASE |
| thurushi | PLEASE | PLESE | PLESSE |
| thurushi | PLEASE | PLES | PLESS |
| thurushi | PLEASE | PLESE | PLESSE |
| saaya | QUEEN | KEN | PKEN |
| saaya | QUIET | QUIET | QUIETT |
| anna | SCHOOL | SCHOOL | SSCHOOL |
| saaya | SCHOOL | SCHOL | SCHOOL |
| saaya | SQUARE | SGAE | SGHAE |
| thurushi | TAXI | T | TI |
| thurushi | WALK | WLX | WALX |
| anna | WEB | FWEB | FEB |
| anna | WEEKEND | WEEKENC | WEEKENCOO |
| saaya | WEEKEND | WEKENK | WEEKENK |
| thurushi | WEEKEND | WEED | WEXED |
| anna | ZERO | FAZZERO | FAZZEROO |
| saaya | ZOO | ZO | ZOO |
