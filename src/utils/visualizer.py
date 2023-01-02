import pandas as pd
import matplotlib.pyplot as plt
from icecream import ic


def to_data_frame(args, pred, label, y_mark):
    pred = pred[0, :, 0]
    label = label[0, -args.pred_len:, 0]
    y_mark = y_mark[0, -args.pred_len:, :]
    y_mark = y_mark.squeeze().tolist()
    df = pd.DataFrame(y_mark, columns=['year', 'month', 'day', 'weekday', 'hour', 'minute'])
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
    df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df = df.drop(['year', 'month', 'day', 'hour', 'minute'], axis=1)

    df['pred'] = pred
    df['label'] = label

    return df


def visualizer(args, predictions):
    # predictions is a list of (output, y, y_mark)
    # and each output and y has shape (batch_size, pred_len, num_features)
    # X_mark and y_mark are the corresponding time stamps
    vis_col_row = args.vis_col_row

    filtered_predictions = [x for i, x in enumerate(predictions) if i % args.seq_len == 0]
    filtered_predictions = filtered_predictions[:vis_col_row ** 2]

    fig = plt.figure(figsize=(16, 5))
    for i, (pred, label, y_mark) in enumerate(filtered_predictions):
        fig.add_subplot(vis_col_row, vis_col_row, i + 1)
        df = to_data_frame(args, pred, label, y_mark)
        plt.plot(df['pred'], label='pred')
        plt.plot(df['label'], label='true')
        plt.legend()
    plt.show()
