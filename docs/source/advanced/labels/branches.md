(target-label-branches)=
# Label branches

Within a single branch, each timepoint can only belong to **one label** ---
overlapping labels in the same branch are trimmed/split automatically.
Branches let you keep **independent, overlapping tiers** (e.g. transient
`song`/`peck`/`jump` events vs. longer `active`/`resting`/`sleep` states)
that never trim each other.

Analogous to **git branches**: only one branch is active (editable) at a
time, and changes you make in one branch can never change labels in another.

```{raw} html
<video autoplay loop muted playsinline style="width:100%">
  <source src="../../_static/media/branch.mp4" type="video/mp4">
</video>
```

---

## Fixed branch positions

There are at most **3 branches**, each with a fixed draw position:

| Branch | Draws as |
|--------|----------|
| `0` | **Full** --- fills the plot |
| `1` | **Top 1** --- thin top strip |
| `2` | **Top 2** --- thin strip below Top 1 |

If Top 1/Top 2 are shown, Full automatically stops short of them instead of
covering them.

Exactly one branch is **active** (editable) at a time. Only the active
branch's labels can be created, selected, deleted (Ctrl+D), or played back
(V) --- labels on other branches are protected even while shown.

- Click a branch's name in the Labels panel to make it active (highlighted).
- Each branch has its own **checkbox** (left of its **x** delete button) to
  show/hide it as an overlay --- independent of which branch is active.
- **Shift+B** swaps the active branch with the previously-active one.
- Drag a label row between branch tables to move it; the mapping file
  updates automatically.
- **+** adds a new branch (max 3); **x** deletes one (must be empty first).
- Imported **predictions** are a separate overlay, toggled with the
  "Predictions" checkbox or **Ctrl+Y**, filling whichever of Top 1/Top 2
  isn't already used by a shown branch.

---

## Assigning branches in `mapping.txt`

Add a third column to any line in `mapping.txt` (values `0`-`2`; omitted
defaults to `0`):

```
0 background
1 song 0
2 peck 0
3 jump 0
4 active 1
5 resting 1
6 sleep 1
```

Here `song`/`peck`/`jump` live on branch 0 (Full) and `active`/`resting`/
`sleep` on branch 1 (Top 1) --- a `song` interval and an `active` interval
can overlap freely. See {doc}`mapping` for the full mapping file format.
