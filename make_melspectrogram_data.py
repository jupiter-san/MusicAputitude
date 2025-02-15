import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 曲をメルスペクトログラムに変換し、画像にして保存するSUB
def melspectrogram_conv(y,sr,s_name):
    # n_mels is number of Mel bands to generate
    n_mels=128
    # hop_length is number of samples between successive frames.
    hop_length=2068
    # n_fft is length of the FFT window
    n_fft=2048
    # Passing through arguments to the Mel filters
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    log_S = librosa.power_to_db(S, ref=np.max)
    #print(log_S.shape)
    plt.figure(figsize=(12, 4),dpi=20)
    librosa.display.specshow(data=log_S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.axis('off')
    plt.subplots_adjust(left=0,right=1,bottom=0, top=1)
    plt.savefig(s_name, format="png")
    plt.close()


dir_name ="music/ARASHI"
data = []

for current_dir,dirs,files in os.walk(dir_name):
    for file in files:
        # mp3データのみ以下の処理を行う
        if file.rsplit(".")[-1] == "mp3":
            file_name = os.path.join(current_dir, file)
            file_noext = os.path.splitext(file)[0]
            new_file_path = os.path.join(current_dir.replace("mp3_data", "image_data")) 

            # 保存ディレクトリが存在しないと書き出し時にエラーになるので、存在確認＆作成
            if not os.path.exists(new_file_path):
                os.makedirs(new_file_path)
        
            offset = 0.0
            i = 1
            # 曲のはじめから60秒ごとに分割。
            # offset 開始秒を60秒ずつずらすことで、60秒ごとにデータを分割
            while True:
                # Load a flac file from 0(s) to 60(s) and resample to 4.41 KHz
                y, sr = librosa.load(file_name, sr=4410, offset=offset, duration=60.0)
                # 最後の60秒未満は捨てる。
                if y.shape[0] / sr < 60: break
                # メルスペクトグラム画像の書き出し
                #　music/{musician}/image_data/ファイル名_01.png　の形式
                s_name = os.path.join(new_file_path, f'{file_noext}_{i:02}.png')
                melspectrogram_conv(y,sr,s_name)
                # 次の分割の読み込み
                offset += 60
                i += 1
                
            # Target の作成
            # (ファイルパス,教師ラベル)をcsvで書き出す
            data.append([file,current_dir.rsplit("/")[-2]])
print(data)  
df = pd.DataFrame(data, columns=['file','label'])
df.to_csv("music_label_ARASHI.csv")

