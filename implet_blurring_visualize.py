import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mode = 'all'

model_names = ['FCN', 'InceptionTime']
xai_names = ['DeepLift', 'GuidedBackprop', 'InputXGradient', 'KernelShap', 'Lime', 'Occlusion',
             'Saliency']

result = pd.read_csv(f'output/blurring_test_{mode}.csv')

plt.figure(dpi=300)

for model_name in model_names:
    for xai_name in xai_names:
        df = result[result['model_name'] == model_name]
        df = df[(df['xai_name'] == xai_name) | (df['xai_name'].isnull())]

        df.loc[df['task_name'] == 'DistalPhalanxOutlineCorrect', 'task_name'] = 'DPOC'

        sns.barplot(df, x='task_name', y='acc_score', hue='method')
        plt.title(f'{model_name}, {xai_name}')
        plt.xticks(rotation=70)
        plt.tight_layout()
        plt.savefig(f'figure/blurring_test_{mode}/{model_name}_{xai_name}.png')
        plt.clf()
