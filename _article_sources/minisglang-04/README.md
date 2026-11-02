# Mini-SGLang Part 4 authoring sources

Final Jekyll draft: `_posts/2026-11-01-minisglang-04.md` (relative to the blog repository).

- `article.md`: platform-neutral article; `article.html` plus `article.assets/`: portable reading view.
- `article.json`: source claim manifest at Mini-SGLang `9a91cfafe754aa85daee49998176275667eb58f2`.
- `evidence.json`: final illustrative example, claims, figures, checks and gaps.
- `assets/draw_figures.py`: regenerate the three editable SVG figures with Python's standard library.
- `assets/check_core.py` and `.txt`: actual CPU contract checks and output.
- `jekyll-profile.json`, `asset-manifest.json`: site adapter configuration and staged asset map.
- `site-overrides.css`: preserve readable figure widths with horizontal scrolling; added to the final post's media style block. The post also adds a short scrolling hint after Figure 1.

The Jekyll adapter output was adapted to use `post_url` for Parts1–3. It was built in a separate preview directory, then copied into the local blog. The blog repository’s release commits record publication state.

## Validation

- Article checker: 0 errors, 18 source claim references checked.
- Nine verbatim Python excerpts match the selected source.
- Independent reviewer checked 34 sampled claims. The final five-page example corrects an admission/capacity issue found during review.
- Actual core CPU checks passed on Python3.10.18 / PyTorch2.7.1: request counters/host appends, context cleanup/nesting, greedy-selection predicate boundaries.
- Staged Jekyll4.3.3 build passed; final local watcher rebuilt the adapted post successfully. HTTP200, local assets and real HTML fragment anchors checked.
- Browser: all three SVGs loaded, desktop page had no document-wide horizontal overflow, figures retain a 760px readable width inside local scroll containers. Inspected all figures and a 390px iframe viewport; exercised mobile figure scrolling.
- GPU attention, actual sampling, CUDA graphs, numerical model parity and TP execution were not run. Those gaps are stated in the article.

## Reproduction

Run `assets/draw_figures.py` to regenerate SVGs. Run `assets/check_core.py` from the pinned Mini-SGLang repository with `PYTHONPATH=python` and a compatible installed PyTorch environment. The script needs no model weights, FlashInfer or GPU.

The article intentionally uses five usable two-token pages (one more than Part3), plus the Engine's dummy page. B's initial slots remain `[0,1,2,3,6,7]`; decode uses `[8,9]`. Token IDs and model dimensions are illustrative.
