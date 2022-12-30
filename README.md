# TODO

<!-- -->

## data folder

- [ ] Complete `crypto_api.py`
    - [ ] BTC 15 minute data
    - [ ] ETH 15 minute data
    - [ ] BTC 1 day data
    - [ ] ETH 1 day data
- [ ] Complete Forex data
    - [Link](https://fred.stlouisfed.org/tags/series?t=&et=&ptic=819202&ob=pv&od=&tg=&tt=)
    - [ ] USD/KRW data
    - [ ] SP500
    - [ ] NASDAQ
    - [ ] DOW
    - [ ] CPI
    - [ ] 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
    - [ ] Federal Funds Effective Rate
    - [ ] University of Michigan: Consumer Sentiment
- [ ] Complet Stock data
    - [ ] AAPL
    - [ ] AMZN
    - [ ] GOOG
    - [ ] MSFT
    - [ ] FB
    - [ ] TSLA
    - [ ] NVDA
    - [ ] NFLX

## dataloader

- SEQ_LEN 과 PRED_LEN에 따라 알맞게 데이터를 불러오도록 해야함
- 원하는 COLUMN 선택하도록 구현
- TRAIN LOADER 구현 
    - Multi Variate와 Univariate를 구분해서 구현
    - SCALING 구현
    - Validation 스플릿 안 겹치게 구현
- PRED LOADER
    - IMS 방식과 DMS 방식을 분리해야함

## model

- Basic LSTM 구현

## train_files

- serial forecasting part
    - train 구현
    - log save directory 구현
        - model name - architecture - version 폴더
        - loss 그래프 tensorboard에서 확인할 수 있도록
        - 하이퍼파라미터들 저장할 수 있도록
    - utils visualizer 에서 test prediction 그래프 그려주는 함수 구현
    - metric diplay 구현

- reinforcement learning part
    - basic DQN 구현
    - MCS 로 EARNING한 것 visualize
    - BACKTESTING 구현

## utils
- 다양한 metric visualization 파트에서 확인할 수 있도록 구현

## train.py
- argparse로 다양한 옵션들을 받을 수 있도록 구현
