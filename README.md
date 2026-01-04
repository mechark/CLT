### Probing

Probing results for GPT2-small:
| layer | label_type | accuracy | precision | recall | f1_score |
|-------|------------|----------|-----------|--------|----------|
| 0     | correct    | 0.856    | 0.855     | 0.870  | 0.862    |
| 0     | shuffled   | 0.483    | 0.502     | 0.436  | 0.467    |
| **1**     | correct    | 0.894    | 0.887     | 0.906  | **0.896**    |
| 1     | shuffled   | 0.553    | 0.573     | 0.450  | 0.504    |
| 2     | correct    | 0.863    | 0.864     | 0.854  | 0.859    |
| 2     | shuffled   | 0.481    | 0.475     | 0.605  | 0.532    |
| 3     | correct    | 0.873    | 0.869     | 0.876  | 0.873    |
| 3     | shuffled   | 0.527    | 0.623     | 0.118  | 0.198    |
| 4     | correct    | 0.881    | 0.889     | 0.868  | 0.878    |
| 4     | shuffled   | 0.451    | 0.467     | 0.770  | 0.581    |
| 5     | correct    | 0.865    | 0.899     | 0.811  | 0.853    |
| 5     | shuffled   | 0.540    | 0.564     | 0.210  | 0.306    |
| 6     | correct    | 0.873    | 0.894     | 0.840  | 0.866    |
| 6     | shuffled   | 0.520    | 0.507     | 0.649  | 0.569    |
| 7     | correct    | 0.859    | 0.870     | 0.846  | 0.858    |
| 7     | shuffled   | 0.537    | 0.693     | 0.148  | 0.243    |
| 8     | correct    | 0.837    | 0.819     | 0.853  | 0.835    |
| 8     | shuffled   | 0.528    | 0.541     | 0.166  | 0.255    |
| 9     | correct    | 0.841    | 0.888     | 0.772  | 0.826    |
| 9     | shuffled   | 0.479    | 0.482     | 0.939  | 0.637    |
| 10    | correct    | 0.831    | 0.848     | 0.820  | 0.834    |
| 10    | shuffled   | 0.509    | 0.694     | 0.088  | 0.156    |
| 11    | correct    | 0.843    | 0.886     | 0.791  | 0.836    |
| 11    | shuffled   | 0.535    | 0.689     | 0.139  | 0.231    |

Link to a dataset used for probing: [link](https://huggingface.co/datasets/mechark/controversial_statements)