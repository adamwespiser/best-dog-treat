# Best Dog Treat

Analyze paired-choice dog treat experiments stored in `data/`.

Each non-empty trial line uses:

```text
A/B :: A
```

That means treat `A` was compared with treat `B`, and `A` won. Blank lines split trials into experiment days.

Hand position is interpreted by trial order inside each day. Odd-numbered trials put the first treat on the right and second treat on the left. Even-numbered trials put the first treat on the left and second treat on the right.

A winner of `X` marks a trial as excluded:

```text
D/A :: X
```

Run the analysis:

```sh
python3 -m pip install -r requirements.txt
python3 analyze_treats.py
```

Useful options:

```sh
python3 analyze_treats.py --bootstrap 0
python3 analyze_treats.py --data-dir data --bootstrap 1000 --seed 7
```

Generate presentation plots:

```sh
python3 plot_treats.py
```

The plots are written as PDFs in `plots/`. If `data/treats.csv` exists, it is used as optional label metadata for the plot text.
