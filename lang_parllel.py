# -*- coding: utf-8 -*-

import speech_recognition as sr
import os
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

path = r'C:\Users\SHIBASIS\Desktop\FreeLancer Works\Speech Recognition\iiit_mar_ash\wav'

def text(filename):
    
    try:
        r = sr.Recognizer()
        with sr.AudioFile(f'{path}\\{filename}') as source:
        # listen for the data (load audio to memory)
            audio_data = r.record(source)
            # recognize (convert from speech to text)
            text = r.recognize_google(audio_data)
            print(text)
    #         telgu.append(text)
            data = pd.DataFrame({'Filename':[filename],'text':text})
            data['language'] = 'Marathi'
            return(data)
    except:
        data = pd.DataFrame({'Filename':['Error'],'text':'Error'})
        data['language'] = 'Error'
        return(data)
        
    
if __name__ == '__main__':
   
    path = r'C:\Users\SHIBASIS\Desktop\FreeLancer Works\Speech Recognition\iiit_mar_ash\wav'
    filenames = os.listdir(f'{path}')

    
    pool = mp.Pool(mp.cpu_count())
      
       # Step 2: `pool.apply` the `howmany_within_range()`
    results = tqdm(pool.map(text,filenames))
    
    # Step 3: Don't forget to close
    pool.close()
    
    results_df = pd.concat(results)
    results_df.to_csv(r'C:\Users\SHIBASIS\Desktop\FreeLancer Works\Speech Recognition\Marathi.csv')
    