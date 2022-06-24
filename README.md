# Examples for Photon ML

Photon ML Framework: https://github.com/sequenzia/photon

---
## Notebooks

- [Deep Ensemble](notebooks/colab/deep_ensemble.ipynb) (Colab Notebook)

---

# Python Scripts

- [Deep Ensemble](scripts) (Python run script and config files)

---
## Datasets

[Sample Market Data](https://storage.googleapis.com/photon-ml-public/data/SPY_1T_2016_2017.parquet): (Apache Arrow Parquet File)
- SPY ETF market data in 1M resolution (2 years: 2016-2017)
- Includes some predefined features that are based of off some standard technical price indicators
- Includes predefined label groups (1,2,3,4,5,6 & 7 days in the future)
  - Discrete price movements with rate of change; used for regression inferences
  - 5 predefined classes for each label group; for used for classification inferences


