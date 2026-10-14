# Dataset attribution

Fanaee-T, H. (2013). Bike Sharing [Dataset].
UCI Machine Learning Repository.
https://doi.org/10.24432/C5W894

License: Creative Commons Attribution 4.0 International.
https://creativecommons.org/licenses/by/4.0/

The dataset contains hourly and daily rental counts from the
Capital Bikeshare system during 2011–2012.

The download script preserves the original CSV and README bytes.
Download metadata and SHA-256 hashes are recorded in
reports/data_manifest.json.

This project will forecast total rental count (`cnt`).
The same-hour `casual` and `registered` counts must not be used
as predictors because their sum equals the target.

Future observed weather will not be supplied as if it were known
at forecast time. Training and evaluation will use chronological
splits.

Before creating lag features, we will audit missing timestamps.
An absent hourly record will not automatically be treated as zero.
