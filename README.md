# Wumpus World 

This is a small console project based on the classic **Wumpus World** problem.

I run a little explorer in a hidden grid. The explorer starts at **(1, 1)** and tries to find the **gold** without falling into a **pit** or walking into the **Wumpus**.

## What I see (percepts)

At each step, the explorer only gets these clues about the *current* cell:

- **Breeze**: at least one adjacent cell has a pit
- **Stench**: the Wumpus is in an adjacent cell
- **Glitter**: the gold is in the current cell

Adjacent means up/down/left/right (no diagonals).

The explorer keeps a small knowledge base:
- cells it has *ruled out* as pits or Wumpus
- cells that are still *maybe* dangerous

Then it tries to move only through cells it currently believes are safe.

## Coordinate system

I enter positions as **x y**.

- **(1, 1)** is the bottom-left corner.
- x increases to the right.
- y increases upwards.

Example: `3 2` means “column 3, row 2”.

## Run it

Requirements: Python 3.9+ (no extra packages)

```bash
python wumpus_game.py
```

The script asks me for:

1. Grid size `N` (for an `N x N` grid)
2. Pit positions (as many as I want, type `done` when finished)
3. One Wumpus position
4. One Gold position

There’s also an optional random seed (press Enter to skip). The seed only affects which safe move the explorer picks when it has choices.

## Example input

If I want a 4×4 world:

- pits at (2,2) and (4,3)
- wumpus at (3,4)
- gold at (4,4)

I can type:

```
Grid size N (for N x N): 4
Random seed (press Enter to skip): 7

PIT: 2 2
PIT: 4 3
PIT: done
WUMPUS (x y): 3 4
GOLD (x y): 4 4
```

## Output legend

### Agent view (during the run)

- `A` = agent position
- `V` = visited cell
- `S` = cell the agent currently believes is safe
- `?` = unknown

### True world (only printed at the end)

- `P` = pit
- `W` = wumpus
- `G` = gold
- `.` = empty

## Notes

This is a “simple logic” explorer, not a perfect solver.  
It plays safe based on the clues it has, and sometimes it will stop early if it can’t justify any safe move.
