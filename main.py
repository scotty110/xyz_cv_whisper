import torch
import numpy as np
import pandas as pd

#from transformers import WhisperFeatureExtractor, WhisperModel
from datasets import load_dataset
from transformers import WhisperProcessor,WhisperForConditionalGeneration
from os.path import join
import glob


class speech_model():
    def __init__(self, device):
        self.device = device
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        self.model.to(self.device)

    def process(self, arrays:np.array) -> str:
        #y = np.load(f_name)
        #inputs = self.processor(arrays, sampling_rate=16000, padding=True, truncation=True, return_tensors="pt")
        inputs = self.processor(arrays, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features
        input_features = input_features.to(self.device)

        generated_ids = self.model.generate(inputs=input_features, max_length=448)
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True) #[0]
        return transcription


def process_dir(f_list:list, model:speech_model):
    max_batch = 128 
    n = len(f_list) // max_batch
    f_list = f_list[:max_batch * n] #Filter down

    df_list = [pd.DataFrame for i in range( n )]
    for i in range(n):
        st = max_batch * i
        batch = [np.load(f) for f in f_list[st:st+max_batch]]
        trans = model.process( batch )
        df = pd.DataFrame( [{'file':f_list[st+i], 'trans_sentence':trans[i][:]} for i in range(max_batch)] )
        #df = pd.DataFrame( [{'trans_sentence':trans[i]} for i in range(max_batch-1)] )
        df_list[i] = df

    all_df = pd.concat(df_list)
    all_df['file'] = all_df['file'].apply(lambda x: (x.split('/')[-1]).split('.')[0] )
    return all_df
        

def read_tsv(f_dir:str):
    files = glob.glob(join(f_dir,'*.tsv')) 
    df_list = [pd.read_csv(f, sep='\t', low_memory=False) for f in files]
    df = pd.concat(df_list)
    return df 


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sm = speech_model(device)

    array_dir = '/home/squirt/Documents/code/convert_voice/cv-corpus-11.0/np_array'
    array_files = glob.glob(join(array_dir,'*.npy')) #[:12]
    df = process_dir( array_files, sm )
    df.to_csv('./processed.csv')
    
    '''
    tsv_dir = '/home/squirt/Documents/code/convert_voice/cv-corpus-11.0/'
    df = read_tsv( tsv_dir )
    print(len(df))
    '''
