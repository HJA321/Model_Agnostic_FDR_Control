# Third-party implementation notes

The following files are sequence-model adaptations used by these experiments:

- `baselines/AAAI-2019-AFS-master/afs_feature_selection.py`: AFS, Gui, Ge and Hu (2019).
- `baselines/alg-net-main/feature_selection_prox.py`: group-lasso/adaptive
  group-lasso proximal selection, associated with Dinh and Ho (2020).
- `baselines/deep-feature-selection-main/proposed/feature_selection_mirror.py`:
  data-splitting mirror feature selection, associated with Sawaya (2025).

These should not be described as unmodified upstream implementations.
The existing MIT license from `deep-feature-selection-main` is preserved at
`baselines/deep-feature-selection-main/LICENSE`. No separate license file was
present in the other two source directories. Their provenance and applicable
permissions need to be confirmed before distributing a fully licensed release.
No project-wide license is inferred from the included third-party license.
