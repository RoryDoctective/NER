# Chinese NER project 

## Entry File

myner.py

withW2V.py



## File structure

- **wiki-corpus** 
  - This folder is complicated, as the original Wikipedia zipped file is 2.23GB large. I did not include them, see the next section for detail. 
- **Dataset**
  - The dataset folder contains all 8 datasets. Datasets found but not used are not included. 
  - The file is named demo.dev, demo.test, and demo.train are pre-processed files
  - The files with other names are the original source files downloaded. 
  - Links to all datasets:
    - MSAR: 
      - https://github.com/yzwww2019/Sighan-2006-NER-dataset
    - Weibo NER:
      - https://github.com/hltcoe/golden-horse
    - Resume:
      - https://github.com/jiesutd/LatticeLSTM/tree/master/ResumeNER
    - Literature:
      - https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset/tree/master/ner
    - CLUENER
      - https://www.cluebenchmarks.com/introduce.html Original 
      - https://github.com/jiachenwestlake/Entity_BERT Processed 
    - Novel & Financial
      - https://github.com/jiachenwestlake/Entity_BERT
    - E-Commerce
      - https://github.com/PhantomGrapes/MultiDigraphNER/tree/master/data/ecommerce

- **Figure**
  - Every time the myner.py or the withW2V.py finished training a model, the figure is saved here to spot overfitting. 
  - It includes some sample pictures. 
- **Presentation**
  - This folder includes the code to plot the visualization of the final results. Basically to generate the figure for the report.
- **Radical**
  - This folder includes:
    - the original CHISE IDE file contains the character to IDCs and DCs information. 
    - the 214 KangXi radicals file
    - the Unihan files contain the character-to-radical one-to-one correspondence. 

- **Report** 
  - Every time the myner.py or the withW2V.py finished training a model, the report is saved here
  - It shows the character, the model predicted tag, and the ground truth tag separated by a tab. 

- **Save_model** 
  - As the name suggests, every time the myner.py or the withW2V.py finished training a model, the model parameters and the model itself are saved here

- **Tune**
  - This folder includes
    - The grid search tuning raw result for each model in .csv format 
    - The .ipynb code to read, process, and draw the data. 

## The wiki corpus 

- I follow the guidance/tutorial from: https://bamtercelboo.github.io/2018/05/10/wikidata_Process/

- The original file named "zhwiki-latest-pages-articles.xml.bz2", can be downloaded here https://dumps.wikimedia.org/zhwiki/latest/.
  - Put it inside this folder. 

- Do not unzip this file, run ``` python wiki_process.py zhwiki-latest-pages-articles.xml.bz2 wiki.txt``` This code extract that .xml file into text files. 
- Next run ```python chinese_t2s.py –input wiki.txt –output simple_ch.txt``` This code translates all traditional Chinese into Simplified Chinese.  
- Next run ```python clean_corpus.py –input simple_ch.txt –output only_ch.txt``` This code only leaves Chinese characters. 
  - This ```only_ch.txt``` file is the 1.21GB training corpus for radical embedding. 
- Finally, run ```python word_to_radical.py –input only_ch.txt –output only_rd.txt```  to translate all Chinese characters into their corresponding radical. 
  - This ```only_rd.txt``` is the 1.21GB training corpus for radical embedding. 
- NOTE! the code ```wiki_process.py, chinese_t2s.py,clean_corpus.py``` are not created by me, but from https://github.com/bamtercelboo/corpus_process_script. I modify and adapt them to get the radical substitution, and created the ```word_to_radical.py```. 

End of the processing 

- the ```view_file.py``` is just for viewing the data because it is huge in size many text editors have problems opening it. 

- the ```w2v.ipynb``` defines the word2vect model, takes in both only_ch.txt and only_rd.txt, and produces pre-trained embeddings. Those files are created by it: (number means the embedding dimensions)
  - pre_trained_char_100_iter5.txt
  - pre_trained_char_150_iter5.txt
  - pre_trained_char_400_iter5.txt
  - pre_trained_char_500_iter5.txt [This one is directly used by the withW2V.py]
  - pre_trained_char_50_iter5.txt
  - pre_trained_rad_100_iter5.txt [This one is directly used by the withW2V.py]
  - pre_trained_rad_25_iter5.txt
  - pre_trained_rad_50_iter5.txt

