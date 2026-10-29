# Development and release status

Poise provides four versioned Sharpa Wave baselines: fixed-palm Hex40, nine
convex objects, random-wrist Hex40 FullPose and fixed-palm HexHammer FullPose.
All reference policies use the 244D actor / 492D critic / 22D action interface
without tactile input. Experimental tactile recipes remain available separately.

Training, simulation rollout and grasp-cache generation have been exercised.
The supported simulator stack is pinned in environments/train/manifest.json.
FoundationPose/control adapters have offline tests; independent native deployment
installation, tracking throughput and physical closed-loop operation are not
validated release features. A new hand requires implementation work.

MIT covers Poise-owned code; third-party
exceptions and separate asset terms are in NOTICE and manifests/rights.json.
Shared asset packages and their GitHub Release URLs are pinned in manifests/releases.
Packaging locally does not change repository visibility or launch training.

See [release preparation](release/README.md), [baselines](baselines/README.md),
[reproduction](reproduction.md) and [validation](release/validation.md).
