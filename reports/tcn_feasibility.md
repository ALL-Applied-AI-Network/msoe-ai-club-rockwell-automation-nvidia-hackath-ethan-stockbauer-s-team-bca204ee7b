# Causal TCN feasibility assessment

Status: **architecture and causal input preparation only; training is not justified yet.**

Video 5 provides one labeled run with 30 vials, ten recipes, three consecutive vials per recipe,
and no verified contamination outcomes. The 149 extracted frames are correlated observations from
those same 30 vials, not 149 independent examples. Random frame- or vial-level splitting would leak
near-duplicates and recipe/run context.

The repository includes left-padded time-ordered sequence construction, missingness masks, a strictly
causal dilated residual TCN architecture, and a causality smoke test available with the optional
`tcn` dependency.

A defensible learned detector needs multiple independent runs containing verified normal and abnormal
outcomes. Splits should be grouped by run or entire recipe-transition groups. Until then, the TCN
must not be trained or described as a contamination classifier. A future self-supervised forecasting
objective could predict the next color-feature vector using normal independent runs, but its anomaly
threshold would still require validation data.
