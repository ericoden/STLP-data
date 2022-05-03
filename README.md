# STLP Data

Data set for the Shared Truckload Problem.

Running:
```console
python3 instance_generator.py C seed
```
will generate a dataframe containing the data for a instance with `C` customers, with the random seed set to `seed`. The dataframe is saved to `data/artificial-custs-[C]-seed-[seed].pkl`.

The dataframe consists of 2 * `C` rows, with the pickup location of customer i in row i, and the delivery location in row i + `C`. The columns are as follows:

* ID
* X coordinate
* Y coordinate
* ID of pickup node
* ID of delivery node
* Demand level
* Load Size (if pickup)
