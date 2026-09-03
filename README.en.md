# Phishing triage by disagreement between classifiers

[![CI](https://github.com/eduolihez/phishing-triage/actions/workflows/ci.yml/badge.svg)](https://github.com/eduolihez/phishing-triage/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB.svg?logo=python&logoColor=white)](requirements.txt)
[![No GPU](https://img.shields.io/badge/GPU-no%20necesaria-brightgreen.svg)](requirements.txt)

### [Versión en Español](README.md) · **[English version](README.en.md)**

An analyst cannot review every message a classifier flags. Can the disagreement
between several classifiers tell them which ones to review first?

This repository measures that question over 7,822 real emails. The answer the
data gives is not the one I expected, and that is the most interesting result
of the work.

![Triage curves and error concentration by decile](resultados/triaje.png)

---

## The result in one line

> Disagreement between classifiers does work for prioritising review
> (AUC 0.905 against 0.515 for chance), but it clearly loses to a much simpler
> signal: the ensemble's own mean uncertainty (AUC 0.959).

To catch 95% of the ensemble's errors you have to review:

| Priority signal | Triage AUC | Mail to review |
|---|---|---|
| Oracle (theoretical ceiling) | 0.991 | 1.75% |
| **Mean uncertainty** (baseline) | **0.959** | **14.10%** |
| Disagreement: standard deviation | 0.905 | 22.62% |
| Disagreement: majority strength | 0.907 | 41.20% |
| Chance (floor) | 0.515 | 97.66% |

The practical headline: reviewing 14% of the mail catches 19 out of every 20
errors, and you don't need an ensemble for that. Looking at where the model is
unsure is enough.

## Why I expected the opposite

The idea comes from two places:

- The [BSC verification models for Catalan](https://huggingface.co/BSC-LT/catalan-verification-model-pkt-a),
  which train two copies of the same ASR on disjoint halves of the corpus so
  their errors are not correlated. If both transcribe the same way, the
  transcription is reliable.
- A [2026 study on medical transcription](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1829902/full)
  that uses disagreement between 8 ASR systems as a *reference-free* signal for
  deciding what a human reviews: it recovers 93.7% of the errors while
  reviewing 28.6% of the text.

The structure of the problem in a SOC is identical: no ground truth, and human
review that is expensive and scarce. It looked transferable. The numbers say it
isn't, at least not here.

## The explanation the data suggests

The ensemble has four views of the same email, and they are not equally strong:

| View | What it looks at | Accuracy | AUC |
|---|---|---|---|
| `remitente` | Semantic headers: who it claims to be | 93.86% | 0.985 |
| `urls` | Where the links go | 94.38% | 0.986 |
| `estructura` | How the HTML is built | 95.14% | 0.980 |
| **`texto`** | **What the message says (TF-IDF)** | **98.17%** | **0.998** |
| *Ensemble* | *Mean of the four* | *98.17%* | *0.999* |

The text view on its own matches the entire ensemble. When one member is far
stronger than the rest, "disagreement" stops measuring genuine doubt and starts
measuring *the weak ones being wrong*, which tells you nothing about whether the
ensemble is right.

That is the difference with the BSC setup: their two models have almost
identical WER (3.85% and 3.74%). Matched strength by design. That seems to be
the condition that makes disagreement useful, and it is a condition that does
not hold here.

Hypothesis for the next experiment: repeat this with four models from the
*same* family trained on disjoint partitions of the corpus, replicating the
BSC method rather than just its intuition, and check whether disagreement
recovers once the strengths are matched.

## The temporal bias, and why it doesn't invalidate the result

Public phishing corpora have a problem the literature carries around and almost
never declares: the legitimate mail and the phishing come from different eras.

| Class | Source | Volume | Era |
|---|---|---|---|
| Phishing | Nazario `phishing0-3.mbox` | 4,572 | 2004-2007 (87%) |
| Legitimate | SpamAssassin `easy_ham` + `hard_ham` | 2,750 | 2002 (100%) |
| Spam | SpamAssassin `spam` | 500 | 2002 |

A classifier can be right by recognising the year rather than the phishing.
That is why `experimentos/02_control_epoca.py` explicitly measures what that
shortcut is worth:

| Test | Result | Interpretation |
|---|---|---|
| Classify using only the year | 99.74% accuracy | The shortcut exists and is nearly perfect |
| Evaluate only on the overlapping band (≤2004) | 98.31%, AUC 1.000 | The model isn't using it: it holds up within the same era |
| Does the text carry an era marker? | AUC 0.943 | The shortcut is available in the vocabulary |

The second row is what saves the experiment: restricting the evaluation to mail
from the same era, performance does not drop. With the caveat that this band
contains only 31 phishing messages, so the evidence is indicative rather than
conclusive.

Measures taken to avoid handing over the shortcut:

- All transport headers (`Received`, `X-Mailer`, `Message-ID`, MIME versions)
  are discarded in `scripts/construir_dataset.py`. Only the semantic ones are
  kept: `From`, `Reply-To`, `Return-Path`, `To`, `Subject`.
- `hard_ham` is included deliberately: legitimate mail that *looks* like spam
  (promotions, heavy HTML). Without it the problem would be artificially easy.
- Spam counts as "not phishing", which forces the model to tell annoying mail
  apart from mail that steals credentials.

## The four views

Error decorrelation is pursued by giving each classifier a different part of
the email, with no access to the others:

```
correo
  ├── remitente    From / Reply-To / Return-Path / To / Subject
  ├── urls         URLs del cuerpo y hrefs del HTML
  ├── texto        asunto + texto visible  (TF-IDF + regresión logística)
  └── estructura   formularios, campos password, iframes, imágenes, ratios
```

A phishing message on a clean domain fools the `urls` view, but `estructura`
sees the password form. That is the complementarity that justifies the setup,
although, as the results show, it isn't enough to make disagreement beat
uncertainty.

## Reproducing it

No GPU. A few minutes on a laptop.

```bash
git clone https://github.com/eduolihez/phishing-triage.git
cd phishing-triage
pip install -r requirements.txt

python scripts/descargar_corpus.py      # descarga Nazario + SpamAssassin
python scripts/construir_dataset.py     # -> datos/corpus.jsonl
python experimentos/01_triaje.py        # resultado principal
python experimentos/02_control_epoca.py # control de sesgo temporal
python experimentos/03_grafica.py       # -> resultados/triaje.png
```

The data is not versioned: it belongs to third parties and the script downloads
it from the original sources, printing SHA-256 fingerprints so you can check
they are the same files.

## Limitations

What this work does not show:

- It does not say disagreement is useless. It says disagreement loses to a
  simpler baseline *in this setup*, with views of unequal strength and over an
  easy corpus (98.2% accuracy, only 43 errors in the test split).
- There are few errors to study. With 43 failures, the triage curves are noisy.
  A harder corpus would give a firmer measurement.
- The corpus is old. Phishing from 2004-2007 looks nothing like today's: no
  widespread HTTPS, no shorteners, no modern domain typosquatting. The
  conclusions about *method* should hold; the absolute figures do not transfer.
- There is no cross-validation. A single 70/30 split with a fixed seed.
  Confidence intervals would come from repeating with several seeds.
- `hard_ham` is only 250 messages, so the hard case is underrepresented.

## Data sources

- [Nazario Phishing Corpus](https://monkey.org/~jose/phishing/), by José Nazario.
- [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/),
  by Apache SpamAssassin.

TREC-07 and CEAS-08 would have been better as legitimate mail, being from 2007
and overlapping in time with the phishing. Both are down today (404 on
`plg.uwaterloo.ca`) and their licence prohibited redistributing any portion.

## License

[MIT](LICENSE). The corpora belong to their respective authors and are not
redistributed here.
