#%%
from icecream import ic
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from orange.src.dataloader.dataloader import create_data_loader
from orange.src.model.NLinear import NLinear, LitNLinear

#%%
"""
   ['target', '0.75_16', '0.5_16', '0.25_16', 'stl_trend',
   'stl_seasonal', 'stl_residual', 'sd_m_trend', 'sd_m_seasonal',
   'sd_m_residual', 'sd_a_trend', 'sd_a_seasonal', 'sd_a_residual',
   'smooth_week', 'smooth_month', 'smooth_2_month', 'smooth_6_month',
   '평균강수량(mm)', '강수일수비율', '평균기온(℃)', '최저기온(℃)', '최고기온(℃)', '평균풍속(m/s)',
   '최대풍속(m/s)', '평균습도(%rh)', '최저습도(%rh)', '일조합(hr)', '일조율(%)',
   '일사합(MJ/m2)', 'holiday', 'sin', 'cos', 't', 'month', 'year', 'covid']
"""

SEQ=365
PRED=365
BATCH_SIZE=32
EPOCHS=200
COLS=   ['0.5_16',
         'sd_m_trend', 'sd_m_seasonal', 'sd_m_residual',
   'smooth_week', 'smooth_month', 'smooth_2_month',
   '평균강수량(mm)', '강수일수비율', '평균기온(℃)',
   'holiday', 'sin', 'cos', 't', 'month', 'year', 'covid']
TARGET='0.75_16'
train_loader = create_data_loader(root='../../orange/data/G_final.csv',
                                  columns_to_use=COLS,
                                  target_column=TARGET,
                                  seq_length=SEQ,
                                  pred_length=PRED,
                                  batch_size=BATCH_SIZE)

val_loader = create_data_loader(root='../../orange/data/G_final.csv',
                                columns_to_use=COLS,
                                target_column=TARGET,
                                seq_length=SEQ,
                                pred_length=PRED,
                                phase='val',
                                batch_size=1,
                                shuffle=False)

test_loader = create_data_loader(root='../../orange/data/G_final.csv',
                                 columns_to_use=COLS,
                                 target_column=TARGET,
                                 seq_length=SEQ,
                                 pred_length=PRED,
                                 phase='test',
                                 batch_size=1,
                                 shuffle=False)

# model
model = LitNLinear(NLinear(seq_len=SEQ,
                           pred_len=PRED,
                           channels=len(COLS)+1),
                           learning_rate=1e-5,
                           weight_decay=1e-5)

# train model
trainer = pl.Trainer(
    log_every_n_steps=300,
    max_epochs=EPOCHS,
    accelerator='mps',
    auto_lr_find=True,
)

trainer.fit(model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            )

mean, scale = train_loader.dataset.means_scales[TARGET]
ic(train_loader.dataset.means_scales[TARGET])
ans = trainer.predict(model=model,
                dataloaders=test_loader)[0].flatten() * scale + mean
sample = pd.read_excel('../../data/Answer_to_June_fixed.xlsx').iloc[:181, 1]

plt.plot(ans, label='pred')
plt.plot(sample, label='label')
plt.legend()
plt.show()

print(mean_absolute_error(ans[:181], sample))