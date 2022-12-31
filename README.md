# TODO

<!-- -->

## data folder

- [x] Complete `crypto_api.py`
    - [x] BTC 15 minute data
    - [x] ETH 15 minute data
    - [x] BTC 1 day data
    - [x] ETH 1 day data
- [x] Complete Forex data
    - [xink](https://fred.stlouisfed.org/tags/series?t=&et=&ptic=819202&ob=pv&od=&tg=&tt=)
    - [x] USD/KRW data
    - [x] SP500
    - [x] NASDAQ
    - [x] DOW
    - [x] CPI
    - [x] 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
    - [x] Federal Funds Effective Rate
    - [x] University of Michigan: Consumer Sentiment
- [ ] Complet Stock data
    - [ ] AAPL
    - [ ] AMZN
    - [ ] GOOG
    - [ ] MSFT
    - [ ] FB
    - [ ] TSLA
    - [ ] NVDA
    - [ ] NFLX

## train_files

- serial forecasting part
    - train 구현
    - `train.py`
        - from src.train_files.serial_forecasting.train_serial에 Experiment 클래스 작성하고 메소드로 train phase predict phase 등 구현
            - expMain 참고
            - predict 하는 함수 직접 짜야함
            - utils visualizer 받아와서 그려주는 것까지
        - argparse로 다양한 옵션들을 받을 수 있도록 구현
            - longExp 참고
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


