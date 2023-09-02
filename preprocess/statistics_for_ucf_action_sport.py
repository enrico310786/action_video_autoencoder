import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

path_df_train = "/home/enrico/Projects/action_video_autoencoder/dataset/ucf_action_grouped_augmented_ttv_mp4/df_train.csv"
path_df_test = "/home/enrico/Projects/action_video_autoencoder/dataset/ucf_action_grouped_augmented_ttv_mp4/df_test.csv"
path_df_val = "/home/enrico/Projects/action_video_autoencoder/dataset/ucf_action_grouped_augmented_ttv_mp4/df_val.csv"
path_df_anomaly = "/home/enrico/Projects/action_video_autoencoder/dataset/ucf_action_grouped_augmented_ttv_mp4/df_anomaly.csv"

df_train = pd.read_csv(path_df_train)
df_test = pd.read_csv(path_df_test)
df_val = pd.read_csv(path_df_val)
df_anomaly = pd.read_csv(path_df_anomaly)

# boxplot
plt.figure(figsize=(25, 12))

print("Number of video for each category: train set")
df_grouped_train = df_train.groupby(['CLASS']).size().reset_index(name='COUNT')
print(df_grouped_train)
plt.subplot(1,4,1)
sns.countplot(x=df_train["CLASS"])
plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
plt.title('Train set clip distribution', fontsize=16)

print("----------------------------------------------")

print("Number of video for each category: test set")
df_grouped_test = df_test.groupby(['CLASS']).size().reset_index(name='COUNT')
print(df_grouped_test)
plt.subplot(1,4,2)
sns.countplot(x=df_test["CLASS"])
plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
plt.title('Test set clip distribution', fontsize=16)
print("----------------------------------------------")


print("Number of video for each category: val set")
df_grouped_val = df_val.groupby(['CLASS']).size().reset_index(name='COUNT')
print(df_grouped_val)
plt.subplot(1,4,3)
sns.countplot(x=df_val["CLASS"])
plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
plt.title('Val set clip distribution', fontsize=16)

print("----------------------------------------------")


print("Number of video for each category: anomaly set")
df_grouped_anomaly = df_anomaly.groupby(['CLASS']).size().reset_index(name='COUNT')
print(df_grouped_anomaly)
plt.subplot(1,4,4)
sns.countplot(x=df_anomaly["CLASS"])
plt.xticks(rotation=30, ha='right', rotation_mode='anchor')
plt.title('Anomaly set clip distribution', fontsize=16)

plt.savefig("/home/enrico/Projects/action_video_autoencoder/preprocess/data_distribution.png")