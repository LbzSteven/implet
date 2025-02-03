import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model_names = ['FCN', 'InceptionTime']
xai_names = ['DeepLift', 'GuidedBackprop', 'InputXGradient', 'KernelShap', 'Lime', 'Occlusion',
             'Saliency']

result = pd.read_csv('output/blurring_test.csv')

plt.figure(dpi=300)

for model_name in model_names:
    for xai_name in xai_names:
        df = result[result['model_name'] == model_name]
        df = df[df['xai_name'] == xai_name]

        df.loc[df['task_name'] == 'DistalPhalanxOutlineCorrect', 'task_name'] = 'DPOC'

        sns.barplot(df, x='task_name', y='acc_score', hue='method')
        plt.title(f'{model_name}, {xai_name}')
        plt.xticks(rotation=70)
        plt.tight_layout()
        plt.savefig(f'figure/blurring_test/{model_name}_{xai_name}.png')
        plt.clf()
