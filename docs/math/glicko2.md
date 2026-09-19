# Glicko-2–inspired player ratings

`pipeline/ratings.py` implements a **Glicko-2-inspired** tracker: each player maintains offensive and defensive means \(\mu\), rating deviations (RD), and simplified volatility, plus rim/perimeter defensive splits.

## State per player

| Field | Meaning |
|-------|---------|
| `O_mu` / `D_mu` | Offensive / defensive rating (default 1500) |
| `D_rim_mu` / `D_peri_mu` | Shot-location defensive splits |
| `O_rd` / `D_rd` | Uncertainty (default RD 350) |
| `O_sigma` / `D_sigma` | Volatility (default 0.06) |

## Update sketch

1. Form expected PPP from \(\mu\) difference / scaling + home boost.  
2. Error = actual PPP − expected PPP (usage-weighted across the stint).  
3. Effective learning rate:

\[
k_{\text{mult}} = \max\!\left(0.5,\; 2\,e^{-n_{\text{games}} / \tau}\right),\quad \tau = \text{K\_MULT\_HALF\_LIFE} = 24.4
\]

\[
k_{\text{effective}} = k_{\text{base}} \cdot \frac{\text{RD}}{350}
\]

\[
\Delta = \text{error} \cdot k_{\text{effective}} \cdot \text{usage\_share} \cdot k_{\text{mult}}
\]

4. Context multipliers from config: clutch boost, garbage-time weight, turnover penalty, foul-draw boost, three-point variance dampening.

## Offseason / inactivity

- `OFFSEASON_REVERSION = 0.15` toward 1500 (current production).  
- Long inactivity inflates RD (uncertainty grows when a player sits).  

## Hierarchical companion

`pipeline/hierarchical.py` blends 1/2/3/5-man combo ratings with base weights \(0.50 / 0.25 / 0.15 / 0.10\) and K decaying by combo size \(\{1, 0.5, 0.25, 0.1\}\).
