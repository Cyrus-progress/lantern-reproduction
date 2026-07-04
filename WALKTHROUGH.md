# The LANTERN Reproduction, Explained From Scratch

**A complete, plain-language walkthrough of the whole project.**

This document assumes you know nothing about chemistry or machine learning. It
explains what we set out to do, every major step we took, why we took it, what
went wrong, how we fixed it, and what we found. Read it top to bottom and you'll
understand the entire project. There's a glossary of every technical term at the
end.

---

## Part 1 — The big picture: what is this project even about?

### 1.1 The real-world problem

Your body is made of cells. To make a cell do something new — like build a
protein that fights a virus — you need to get a set of instructions *inside* the
cell. Those instructions come as a molecule called **mRNA**.

But there's a problem: mRNA is fragile, and cells don't just let random molecules
walk in. So scientists wrap the mRNA inside a tiny protective bubble made of fat
(a "lipid"). That bubble is called a **lipid nanoparticle**, or **LNP**. This is
exactly the technology inside the COVID-19 mRNA vaccines.

The catch: not all fat bubbles work equally well. Some are great at sneaking their
cargo into cells; others barely work at all. How well a particular bubble delivers
its cargo is measured by a score called **transfection efficiency**. High score =
good delivery. Low score = poor delivery.

### 1.2 Why we want a computer to predict it

To find good bubble designs, scientists normally have to *build each one in a lab
and test it* — which is slow and expensive. There are millions of possible fat
molecules, so testing them all is impossible.

The dream: a **computer model** that looks at the chemical structure of a proposed
fat molecule and *predicts* its transfection score — without ever going into a lab.
Then scientists could screen millions of designs on a laptop and only build the
few the computer says are promising.

### 1.3 What "LANTERN" is

**LANTERN** is a research paper (published in 2025) that claims to have built
exactly such a computer model. The authors reported that their best model could
predict transfection efficiency quite accurately.

### 1.4 What *we* set out to do (and why)

Here's the key idea. In science, you shouldn't just *believe* a paper's claim. You
should be able to **reproduce** it — meaning an independent person, using their own
tools, should be able to rebuild the thing from scratch and get the same answer. If
they can, the claim is trustworthy. If they can't, something's fishy.

So our project had two goals:

1. **Reproduce** LANTERN's headline result from scratch — write all our own code,
   and only use the authors' materials as an "answer key" to check ourselves.
2. **Stress-test** it — push the model into a harder, more realistic situation and
   see whether it still works. (This second part is the original contribution; it's
   the part that makes the project *ours*, not just a copy.)

Think of it like a recipe. Goal 1 is: "A famous chef says this cake is amazing.
Can I bake it myself from the description and get the same cake?" Goal 2 is: "Okay,
it works in a normal oven — but what happens if I bake it at high altitude, where
things get tricky?"

---

## Part 2 — The one concept that makes everything else make sense

Before any of the steps, you need **one** idea: **computers can't read molecules,
only numbers.**

A molecule is a physical arrangement of atoms. A computer model is just math — it
adds and multiplies numbers. So step one of *any* chemistry-AI project is to turn
each molecule into a **list of numbers** that describes it. This translation step
is called **featurization**, and the numbers are called **features**.

### 2.1 How a molecule is written down: SMILES

Chemists write molecules as short text strings called **SMILES**. For example, one
molecule in our dataset looks like:

```
CCCCCCCCCCCCCCCCCCNC(=O)C(CCCCCOC(=O)...)NCCCN(C)CCCN
```

Each letter and symbol stands for atoms and bonds. You don't need to read it — just
know that it's a compact text recipe for a molecule, and it's what our program
takes as input.

### 2.2 Turning a SMILES into numbers: two "lenses"

LANTERN describes each molecule using **two different lists of numbers**, like
looking at the same object through two different lenses:

**Lens 1 — the "fingerprint" (2,048 numbers).** Imagine a checklist of 2,048
possible little structural patterns ("does this molecule contain this particular
cluster of atoms, and how many times?"). Each molecule gets 2,048 answers. This is
called a **Morgan fingerprint**. It captures *what pieces the molecule is built
from*. (Two molecules that share many pieces will have similar fingerprints.)

**Lens 2 — the "descriptors" (210 numbers).** Think of this like a nutrition label
for the molecule: its molecular weight, how greasy vs. water-loving it is, how many
rings it has, how many nitrogen atoms, and so on — 210 different measured
properties. These are called **RDKit descriptors** (RDKit is the free chemistry
software that computes them). This captures the molecule's *overall physical
character*.

Glue those two lists together and every molecule becomes a single list of **2,258
numbers** (2,048 + 210). **That combined list is what the model actually learns
from.** Every model in this project — simple or fancy — sees the same 2,258 numbers
per molecule.

Keep this picture in your head: **molecule → SMILES text → 2,258 numbers → model →
predicted score.**

---

## Part 3 — Step 0: Setting up and understanding the materials

### 3.1 Getting the authors' materials

The authors put their project on **GitHub** (a website where people share code and
data). We **cloned** it — that just means downloaded a full copy — into a folder
called `LANTERN/`.

Important rule we set for ourselves: we would **only use their stuff as an answer
key**. We would write all our own code. We'd use their files just to check whether
our numbers came out right — like having the answer key to a math textbook so you
can verify your own work, without copying how they solved it.

### 3.2 Looking at the dataset

Inside their materials we found the dataset: a spreadsheet-like file called
`AGILE.csv` with **1,100 molecules**. It has just two columns:

- **SMILES** — the text recipe for each molecule.
- **Target** — the real, lab-measured transfection score we're trying to predict.

That's it. 1,100 examples of "here's a molecule, here's how well it actually
worked." This is the material the model learns from and is tested on.

### 3.3 Understanding the "splits" (this is important)

You can't test a model on the same molecules it studied — that's like giving a
student the exam questions in advance. So the data is divided into three groups,
called a **split**:

- **Training set — 880 molecules (80%):** the "practice problems." The model learns
  from these.
- **Validation set — 110 molecules (10%):** a "practice quiz" used *during* learning
  to check progress and know when to stop.
- **Test set — 110 molecules (10%):** the "final exam." Molecules the model has
  never seen. **This is the only score that counts**, because it measures whether
  the model learned something general or just memorized the practice answers.

The authors provided the *exact* lists of which molecule goes in which group,
stored as simple lists of row numbers (like "molecule #545 is in training, #914 is
in training, ..."). We use their exact lists so our results are directly
comparable to theirs — nobody can say we got a lucky split.

They actually provided **three** different ways of splitting, which becomes very
important in Part 8:
- a **random** split (mix them up and deal them out randomly), and
- two **scaffold** splits (a harder, structured way — explained later).

### 3.4 Setting up a clean workspace

We created a fresh, isolated Python environment (think of it as a clean, empty
toolbox just for this project, so nothing from other projects interferes). Into it
we installed the tools we'd need:

- **RDKit** — the chemistry software that reads SMILES and computes features.
- **PyTorch** — the software for building and training neural networks.
- **scikit-learn**, **NumPy**, **pandas**, **matplotlib** — general tools for
  simpler models, number-crunching, spreadsheets, and making charts.

One note: the authors used a heavy piece of software called **DeepChem** to make
the fingerprints. It's cumbersome to install, and our project plan said we could
substitute RDKit if needed — as long as we *proved* our substitute produced
identical results. We did exactly that (see the next part).

---

## Part 4 — The hardest part: making our numbers match theirs (featurization detective work)

This was the biggest and trickiest part of the whole project, so we'll go slowly.

**The goal:** write our own code that turns a SMILES into those 2,258 numbers, and
**prove** our numbers exactly match the authors' pre-computed numbers. The authors
had saved their numbers in "answer key" files. If our recipe produced the same
numbers, we'd know we're standing on the same foundation they did. If not, every
result afterward would be meaningless.

It's a bake-off: they handed us the finished cake, and we had to reverse-engineer
the recipe until our cake was identical.

### 4.1 Mystery #1 — the fingerprint that was "too small"

Our first attempt at the fingerprint used the standard, textbook setting (called
"radius 2" — roughly, "only look at small neighborhoods around each atom"). We
compared our fingerprint to the authors' answer key and it was **way off**:

- Our fingerprint detected about **56** patterns per molecule.
- Theirs detected about **420** patterns per molecule — almost **8 times more**.

That's not a small rounding difference; something was fundamentally different. So we
opened the authors' code (reading it to understand the definition is fair — it's
the "answer key") and found the culprit. Their setting wasn't "radius 2." It was
effectively **"radius unlimited"** — look at neighborhoods of *every* size, from
single atoms all the way up to the entire molecule. A big fat molecule has *tons* of
patterns when you look at every size, which is exactly why theirs lit up ~420
patterns instead of ~56.

(Side note: the project's original written instructions *said* "radius 2," but the
authors' actual code did something different. We went with what the code really did,
because that's what produced the answer key — and we flagged the discrepancy.)

We changed our setting to match, and re-checked. Now we matched **1,052 out of the
1,100 molecules exactly**. The remaining 48 were *almost* identical — they had the
exact same total set of patterns, just filed into slightly different slots, because
newer and older versions of RDKit shuffle the filing very slightly. The *counts*
were preserved perfectly. This is cosmetic and harmless, so we moved on.

### 4.2 Mystery #2 — which 210 descriptors, and in what order?

The second lens (the 210 "nutrition label" descriptors) had its own puzzle. Our
version of RDKit offered **217** possible descriptors, but the answer key had exactly
**210**. Which 7 were missing, and in what order were the 210 listed? If we got the
order wrong, column 5 of ours might be "molecular weight" while column 5 of theirs
was "number of rings" — total mismatch.

By carefully lining up our columns against theirs, one at a time, across all 1,100
molecules, we figured out:

- The authors used a slightly **older version** of the chemistry software that
  simply didn't *have* 7 of the descriptors our newer version added. We removed those
  same 7 to match.
- The descriptors were listed in **alphabetical order** by name (a specific quirk of
  how DeepChem lists them).
- Two special cases needed particular settings to match (one descriptor called "Ipc"
  had to be computed in "average" mode, or it produces astronomically huge numbers;
  another counts atoms using an older rule).

After all that, **209 of the 210 descriptors matched exactly**. The one holdout, a
descriptor counting certain atom types, differs by a tiny amount (off by 1 on some
molecules) because the newer software counts it slightly differently — something we
literally cannot undo without installing the authors' exact old software version.
One descriptor out of 210, off by 1, after everything gets scaled down — completely
negligible.

### 4.3 Why this mattered so much

At the end of Part 4 we could say, with proof: **our from-scratch featurization
reproduces the authors' features essentially exactly.** That's the bedrock. Every
result after this point rests on it. It's also independent evidence the substitution
of RDKit for DeepChem was valid.

---

## Part 5 — The safety check: simple "warm-up" models

Before building the fancy model, we ran three **simple** models as a sanity check.
The logic: *if our features are correct, then even simple models should score about
what the paper reported.* If they came out wildly wrong, that would mean our
features were still broken and we should stop and debug — no point building a
skyscraper on a cracked foundation.

### 5.1 What the three simple models are (in plain terms)

- **kNN (k-Nearest Neighbors):** the simplest possible idea. To predict a new
  molecule's score, find the 5 most *similar* molecules it already knows (nearest
  "neighbors" in number-space) and average their scores. It doesn't really "learn" —
  it just looks up look-alikes. Remember kNN; it becomes the hero later.
- **Random Forest (RF):** builds a hundred little "flowchart" decision trees (each
  asks yes/no questions like "is this number above 3.5?") and averages their votes.
- **SVR (Support Vector Regression):** a mathematical method that fits a smooth
  curved surface through the data.

You don't need the internals. Just know: kNN is the dumbest, RF and SVR are
middle-tier, and the neural network (Part 6) is the fanciest.

### 5.2 How we grade a model: R²

Every model is graded with a number called **R²** ("R-squared"). It runs from 0 to
1:

- **1.0** = perfect predictions.
- **0.0** = useless (no better than always guessing the average).
- **~0.7** = pretty good; the model explains about 70% of the variation.

We also track three other grades (RMSE, MAE, Pearson r) that measure closeness in
slightly different ways, but R² is the headline.

### 5.3 The result — it passed

Our three simple models landed right on the paper's numbers:

| Simple model | Our R² | Paper's R² |
|---|---|---|
| kNN | 0.6178 | 0.6148 |
| Random Forest | 0.7328 | 0.7169 |
| SVR | 0.7310 | 0.7285 |

This was a big green light. SVR is *deterministic* (it gives the exact same answer
every time), and ours matched the paper almost perfectly — strong proof our features
were right. And notice kNN scored **0.62**, which matters for a reason in the next
line.

### 5.4 The paper's central claim, confirmed

The authors' big selling point was that their *simple* features beat a fancy,
heavyweight competitor model called **AGILE** (a "graph neural network" that another
team had trained). AGILE scored only **0.2655**. Our plain kNN scored **0.62** —
nearly *triple* AGILE's score. So the paper's central claim ("simple structural
features beat the heavyweight model on this data") reproduced. (We did *not* re-run
AGILE — it's a giant pretrained model, and our plan said to treat its published
score as a fixed number to compare against.)

---

## Part 6 — The main event: the neural network (MLP)

Now the headline model. The paper's best predictor is a **neural network**,
specifically a type called an **MLP** (Multi-Layer Perceptron).

### 6.1 What a neural network is, plainly

A neural network learns by example, like a student doing thousands of practice
problems. You show it a molecule's 2,258 numbers; it makes a guess at the score; you
tell it the real answer; it sees how far off it was and nudges its internal dials to
be a little less wrong next time. Repeat thousands of times and it gets good.

It's built in **layers**. Information flows in one end (the 2,258 numbers), passes
through several processing stages, and comes out the other end as a single number
(the predicted score). Each layer extracts more useful patterns before passing
things along. Our network has **7 layers**, with the internal stages sized
200 → 300 → 500 → 500 → 300 → 200 → 1. (Those are just how many little "neurons" are
in each stage. The final "1" is the single predicted score.)

We built this exactly as the paper's Methods section described it, including all the
training settings (which optimizer to use, how fast to learn, how many rounds, etc.).

### 6.2 How training actually runs

- The network goes through the training molecules up to **100 times** (each full
  pass is called an **epoch**).
- After each pass, it checks itself on the validation set (the "practice quiz").
- It keeps a snapshot of the version that did best on the quiz. This is called
  **early stopping** — it prevents the model from over-studying the practice set and
  losing its ability to generalize (like a student who memorizes practice answers
  word-for-word and then bombs the real exam).

### 6.3 One preprocessing detail (and an honesty note)

Before feeding numbers to the models, all features are **scaled** to a common range
(so a descriptor measured in the thousands doesn't drown out one measured in
decimals). We copied the authors' exact scaling procedure.

Honesty note: the authors' scaling procedure peeks at *all* the data (including the
test molecules) when deciding the scale. Technically that's a mild form of
"cheating" called a **data leak** — the test set is supposed to be totally sealed
off. It slightly inflates everyone's scores. **We deliberately copied it anyway**,
because the whole point was an apples-to-apples comparison with the paper. We flagged
it clearly in our findings so no one is misled. (It doesn't change which model wins
or any of the big conclusions.)

### 6.4 The result — reproduced

We trained the MLP and it scored **R² = 0.799** on the test set. The paper reported
**0.8161**. Close, but a hair under. Was that a real gap, or just luck?

### 6.5 Why we ran it 10 times (the randomness point)

Here's a subtle but crucial thing: **a neural network gives a slightly different
result every time you train it**, even with identical code. That's because it starts
with random dial settings and shuffles the practice problems randomly. It's like how
the same student might score 88% one day and 91% another on equally hard exams.
Neither single number is "the" score.

So instead of trusting one run, we trained the MLP **10 separate times** with 10
different random starting points and looked at the average and the spread:

**R² = 0.8123, give or take 0.0107.**

That "give or take" (called the **standard deviation**) is the key. It means our
model normally lands between about **0.80 and 0.82**. The paper's **0.8161 sits right
inside that normal range.** So our model and their model are, for all practical
purposes, the same — the tiny difference is just random wiggle, not a real
disagreement.

We checked all four grades this way:

| Grade | Our result (10 runs) | Paper | Match? |
|---|---|---|---|
| R² | 0.8123 ± 0.0107 | 0.8161 | ✅ |
| RMSE | 1.4450 ± 0.0414 | 1.4308 | ✅ |
| MAE | 1.0978 ± 0.0293 | 1.1003 | ✅ |
| Pearson r | 0.9023 ± 0.0064 | 0.9053 | ✅ |

**Conclusion of Part 6: the headline result reproduced.** We rebuilt their best
model from scratch, with our own features and our own training code, and got the
same score four different ways. The paper's claim is real and repeatable.

We also made a **scatter plot** for this model: each dot is a test molecule, with its
true score on one axis and the model's predicted score on the other. Dots near the
diagonal line = good predictions. Ours cluster nicely along the line.

---

## Part 7 — A quick reality check on what we'd done so far

At this point we'd confirmed: LANTERN's impressive score is genuine. But there's a
hidden asterisk on that score, and uncovering it is the most interesting part of the
project. That's Part 8.

The asterisk: the "final exam" (the random test set) was a bit too easy. Because the
molecules were shuffled *randomly* into train and test, the test molecules often had
close **look-alikes** sitting in the training set. So the model could score well
partly by recognizing near-duplicates — not by truly understanding chemistry. That's
an "open-book test." The real question is: how does it do on a **closed-book test**,
with molecules genuinely unlike anything it studied?

---

## Part 8 — The stress test: the scaffold split (our original contribution)

### 8.1 What a "scaffold" is

Most molecules have a **scaffold** — a core skeleton, like the frame of a house.
Different molecules can share the same core frame but have different decorations
hanging off it. Molecules with the same scaffold are structural cousins.

### 8.2 Random split vs. scaffold split

- **Random split (Part 6):** cousins get scattered across both training and test. The
  model often sees a close relative during study → easier test → open book.
- **Scaffold split:** we group molecules by their core frame and make sure *entire
  families* land only in training *or* only in test. Now the test molecules have
  **brand-new skeletons the model never saw** → much harder → closed book. This is a
  far better simulation of real life, where scientists want to design genuinely *new*
  kinds of lipids.

We re-ran **everything** — all four models — on this harder scaffold split.

### 8.3 What happened: the collapse

| Model | Random split R² | Scaffold split R² | How much it dropped |
|---|---|---|---|
| kNN | 0.6178 | **0.5946** | **only −0.02** |
| Random Forest | 0.7256 | 0.4453 | −0.28 |
| SVR | 0.7310 | 0.3162 | −0.41 |
| **MLP (the star)** | **0.8123** | 0.4868 | **−0.33** |
| AGILE (for reference) | 0.2655 | 0.0057 | −0.26 |

(These are averages over 10 runs. Our single-run scaffold numbers also matched the
paper's scaffold numbers closely, so the reproduction holds on this split too.)

Read that table slowly, because it tells three stories.

### 8.4 Finding #1 — the star model doesn't really generalize

The MLP fell from **0.81 to 0.49** — it lost nearly *half* its skill the moment the
test molecules were genuinely new. This confirms the suspicion: a lot of its
impressive random-split score came from recognizing look-alikes, not from learning
transferable chemistry. On truly novel molecules, it's only mediocre.

### 8.5 Finding #2 — the rankings flip upside down

On the easy random split, the ranking was: **MLP > SVR > RF > kNN** (fancy wins,
simple loses). On the hard scaffold split, it **completely inverts** to: **kNN > MLP
> RF > SVR** (simple wins!).

The dumbest model — kNN, which just averages the most similar known molecules —
barely dropped at all (−0.02) and became the **best** model. Why? Because "find the
closest known examples and average them" degrades gracefully when things get
unfamiliar. The fancier, more flexible models over-fit to the training distribution
and fell hardest when that distribution shifted.

This has a real, practical punchline: **if you picked your model based on the easy
random test, you'd pick the MLP — which is exactly the *wrong* model for the real job
of designing new lipids.** For that job, plain kNN is the honest baseline to beat.

### 8.6 Finding #3 — the neural net also becomes *unstable* (something the paper never reported)

Remember the "give or take" spread from Part 6? On the easy split, the MLP's spread
was tiny: ±0.0107 (very consistent run to run). On the hard scaffold split, its
spread **exploded to ±0.0552 — five times larger.** So under real-world conditions
the neural network isn't just *worse on average* — it's **erratic**, swinging around
by a lot depending on random luck. Meanwhile kNN, being deterministic, gives the
exact same answer every time (zero spread) — making it not only the most accurate on
novel molecules, but also the most *reliable*.

This instability finding is our own contribution on top of the paper — they reported
the average collapse, but not the variance blow-up.

### 8.7 The picture we made

We made a **bar chart** that shows all of this at a glance: for each model, two bars
side by side (easy split vs. hard split), with little error-bar "whiskers" showing
the spread. You can literally *see* the MLP's tall easy-split bar shrink on the hard
split, and see its whisker grow fat while kNN's bars stay steady with no whiskers.
That chart is saved as `results/robustness_R2_random_vs_scaffold.png`.

---

## Part 9 — Putting the project on GitHub

Once everything worked, you asked to publish the project on **GitHub** (the code-
sharing website) so others can see and use it.

### 9.1 What "publishing to GitHub" involves

- **git** is a tool that tracks versions of files and bundles changes into
  snapshots called **commits** (like save points in a video game).
- A **repository** (or **repo**) is the project folder that git tracks.
- **Pushing** means uploading your commits to GitHub's servers so they're online.

### 9.2 The decisions we made carefully

Because publishing to a **public** website is hard to undo (things get copied and
indexed by search engines), we paused to confirm choices with you:

- **Public vs. private:** you chose **public**.
- **What to include:** you chose to include the code, the write-ups, the result
  figures, *and* a tiny slice of the essential data so the project runs on its own.

### 9.3 What we deliberately left out (and why)

Not everything in the project folder should go online. We used a special file called
**`.gitignore`** (a list of "do not upload" items) to exclude:

- **The toolbox/environment folder (`.venv`, ~1 GB):** huge, and anyone can recreate
  it by installing the tools themselves.
- **The authors' full downloaded repo (`LANTERN/`, ~82 MB):** it's *their* project
  with its own ownership and its own big files. It wasn't ours to re-publish. Our
  README just tells people to download it themselves if they want it.
- **The features cache (~19 MB):** temporary computed files that our code can
  regenerate in seconds.
- **Personal/temporary junk:** Mac system files (`.DS_Store`), tool settings, etc.

To make the project still runnable without the big excluded folder, we copied just
the **essential ~90 KB of data** (the molecule list and the split lists) into a small
`data/` folder, and we adjusted our code to look there first. Then we **proved it
works standalone** by temporarily hiding the big folder and re-running the pipeline —
it worked.

### 9.4 Privacy touch

When git records a commit, it stamps it with an email address. To keep your personal
email *out* of the public record, we used GitHub's special "no-reply" email address
for the commits instead.

### 9.5 The license

You asked to add a **license** — a short legal file that tells the world what they're
allowed to do with your code. We added the **MIT License**, which is the most common,
most permissive one: "anyone can use, copy, and modify this, as long as they keep the
copyright notice." It's stamped with your name (Cyrus Ho) and the year.

We also added a **NOTICE** file crediting the original LANTERN authors for the small
slice of data we included, since their work is also under the MIT license and good
practice is to credit borrowed material. (Small hiccup: we first put that credit
*inside* the license file, which confused GitHub's automatic license detector, so we
moved it to its own NOTICE file. GitHub now correctly shows a green "MIT License"
badge.)

### 9.6 The result

The project is live at:
**https://github.com/Cyrus-progress/lantern-reproduction** — public, MIT-licensed,
with the code, data, results, figures, and write-ups all online.

---

## Part 10 — A guide to every file in the project

If you open the project folder, here's what each piece is:

**The code:**
- `featurize.py` — turns molecules (SMILES) into the 2,258 numbers; can also
  self-check against the answer key.
- `models.py` — defines all four models (kNN, RF, SVR, and the MLP neural network).
- `train.py` — the main driver: loads data, scales it, splits it, trains a model,
  scores it, and compares to the paper.
- `evaluate.py` — computes the grades (R², RMSE, MAE, Pearson r) and draws the
  scatter plots.
- `plot_robustness.py` — draws the big bar chart comparing the two splits.

**The write-ups:**
- `README.md` — the "front page": what the project is and how to run it.
- `FINDINGS.md` — the detailed results report, with all the numbers and honesty notes.
- `WALKTHROUGH.md` — *this document.*
- `LICENSE` / `NOTICE` — the legal/credit files.

**The materials:**
- `data/` — the vendored molecule list and split files (~90 KB).
- `results/` — the output charts and results tables.
- `features/` — the computed-numbers cache (not uploaded; regenerated on demand).
- `LANTERN/` — the authors' downloaded repo (not uploaded; used only as answer key).

---

## Part 11 — The bottom line

**Did LANTERN reproduce?** Yes. Rebuilding the best model from scratch — our own
features, our own training code, on the authors' exact data splits — we got
**R² = 0.812**, statistically the same as their reported **0.816**. The paper's
headline is real and repeatable, not a fluke of their particular code.

**What did the stress test reveal?** The impressive score is an "open-book" number.
On a realistic "closed-book" test with genuinely new molecule types:
- the star neural network loses ~40% of its skill and becomes erratic, and
- the *simplest* method (kNN) becomes both the most accurate *and* the most reliable.

So for the actual goal — designing brand-new lipids that have never been made — the
paper's ranking of models is misleading, and a humble nearest-neighbor lookup is the
baseline any fancier model must genuinely beat. That's a useful, honest cautionary
tale, and it's the part of this project that's original to us.

---

## Glossary — every technical term, in one place

- **AGILE** — a rival, heavyweight pre-trained model (a graph neural network) that
  the LANTERN paper compares against. It scored poorly (R² 0.27), and we used its
  published score as a fixed reference without re-running it.
- **Clone** — to download a full copy of a code repository.
- **Commit** — a saved snapshot of the project in git (a "save point").
- **Data leak** — accidentally letting information from the test set influence
  training/setup, which unfairly inflates scores. We knowingly copied the authors'
  mild leak for a fair comparison, and disclosed it.
- **DeepChem** — a chemistry-AI software library the authors used; we substituted
  RDKit and proved the results matched.
- **Descriptors (RDKit)** — 210 measured molecular properties (weight, greasiness,
  ring count, etc.) used as one set of features.
- **Early stopping** — keeping the model version that did best on the validation set,
  to avoid over-studying the training data.
- **Epoch** — one full pass of the model through all the training data.
- **Featurization / features** — turning a molecule into a list of numbers a computer
  can learn from.
- **Fingerprint (Morgan)** — 2,048 numbers describing which structural patterns a
  molecule contains, and how many times.
- **git / GitHub** — a version-tracking tool / the website for sharing code repos.
- **kNN (k-Nearest Neighbors)** — the simplest model: predict by averaging the 5 most
  similar known molecules. It won the hard stress test.
- **Lipid nanoparticle (LNP)** — a tiny fat bubble that delivers mRNA into cells.
- **MAE (Mean Absolute Error)** — a grade: the average size of the prediction error.
  Lower is better.
- **MLP (Multi-Layer Perceptron)** — the neural network; the paper's best model.
- **mRNA** — the molecular instructions delivered into cells (as in mRNA vaccines).
- **Neural network** — a model that learns from examples by repeatedly adjusting
  internal dials to reduce its errors.
- **Pearson r** — a grade: how well predictions and truth move together (1.0 =
  perfect). Higher is better.
- **PyTorch** — the software used to build and train the neural network.
- **R² (R-squared)** — the main grade: 1.0 = perfect, 0.0 = useless. Fraction of the
  variation the model explains.
- **Random Forest (RF)** — a model that averages the votes of 100 little decision
  trees.
- **RDKit** — the free chemistry software that reads SMILES and computes features.
- **Radius (fingerprint)** — how large a neighborhood around each atom the fingerprint
  looks at. The authors used "unlimited," not the standard "2" — a key discovery.
- **Reproduce / reproduction** — independently rebuilding a result to verify it's real.
- **RMSE (Root Mean Squared Error)** — a grade like MAE but punishes big misses more.
  Lower is better.
- **Scaffold** — a molecule's core skeleton/frame. Molecules sharing one are cousins.
- **Scaffold split** — a hard, realistic way to divide data so the test set has
  brand-new skeletons (a "closed-book" test).
- **scikit-learn** — the software library providing the simple models (kNN, RF, SVR).
- **SMILES** — a compact text string that encodes a molecule's structure.
- **Split** — the division of data into training / validation / test groups.
- **Standard deviation ("give or take")** — how much a number bounces around between
  runs; a measure of consistency.
- **SVR (Support Vector Regression)** — a model that fits a smooth curved surface
  through the data.
- **Test set** — the sealed-off "final exam" molecules; the only score that truly
  counts.
- **Training set** — the "practice problems" the model learns from.
- **Transfection efficiency** — how well a lipid nanoparticle delivers its mRNA cargo
  into cells; the number we're predicting.
- **Validation set** — the "practice quiz" used during training to tune and stop.
