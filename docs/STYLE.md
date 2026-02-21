# Style

Plotting conventions for figures and analysis. Clean, professional plots
following Edward Tufte's philosophy (data-first, minimal decoration)
without being dogmatically spartan.

## Color Palette

| Color | Hex | Role |
|-------|-----|------|
| Crimson | `#DC143C` | Primary data / ETD |
| Steel Blue | `#4682B4` | Secondary data / baseline |
| Slate Gray | `#708090` | Contour lines, tertiary, neutral |
| Teal | `#2E8B8B` | Accent (critical thresholds, reference) |

### Sequential (crimson ramp)

```python
crimson_seq = ['#F5C6CB', '#E89DA3', '#DC143C', '#A10E2B', '#6B0A1D']
```

### Multi-algorithm

| Variant | Color |
|---------|-------|
| ETD-B (primary) | `#DC143C` Crimson |
| ETD-UB | `#A10E2B` Dark Crimson |
| ETD-B-Maha | `#E89DA3` Light Crimson |
| SVGD | `#4682B4` Steel Blue |
| ULA | `#708090` Slate Gray |
| MPPI | `#2E8B8B` Teal |

## Contour Plots

### Background

- **No fill by default.** If fill is used, keep it very light: custom
  colormap from `#FFFFFF` to `#D0D4D8`, `alpha=0.25–0.35`, ≤5 levels.
- **Contour lines:** Slate Gray `#708090`, `linewidths=1.0`, `alpha=0.9`,
  5–7 levels.
- **Contour labels:** Inline via `clabel`, `fontsize=7`, `fmt='%.1f'`,
  color `#607080`.
- **Accent contours** (zero-level, thresholds): Teal `#2E8B8B`,
  `linewidths=1.5`, `linestyles='--'`.
- **Background:** White (`ax.set_facecolor('white')`).

### Overlaid Data

- Markers with **white edges** (`edgecolors='white'`, `linewidths=0.7`).
- Marker size ~60–70 (`s=65`).
- Always `zorder=5` to draw above contours.

### General

- Data is the primary visual element; contours provide context only.
- Legend: `framealpha=0.9`, `edgecolor='#DDDDDD'`.
- Minimize contour levels to avoid clutter.

## Line Plots (convergence, diagnostics)

- Solid lines for primary methods, dashed for baselines.
- Error bands: shaded region (mean ± 1 std), `alpha=0.15`.
- No unnecessary gridlines; light gray (`#E8E8E8`) if needed.
- Log scale on y-axis for energy distance and similar metrics.
- X-axis: iteration count. Label: "Iteration" or "Step."

## Scatter Plots

- Particle scatter: colored by algorithm, white edge, `s=40–65`.
- Reference samples (NUTS): small gray dots, `s=8`, `alpha=0.3`,
  `zorder=1`.

## General

- NeurIPS column width: 3.25 in. Full width: 6.75 in.
- Font: matplotlib default (DejaVu Sans) or Times if matching LaTeX.
- Font sizes: axis labels 9pt, tick labels 8pt, legend 8pt, title 10pt.
- Tight layout: `plt.tight_layout(pad=0.5)`.
- Save as PDF for vector quality: `savefig('fig.pdf', bbox_inches='tight')`.
