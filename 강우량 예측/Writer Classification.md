```python
import pandas as pd
import warnings 
warnings.filterwarnings(action='ignore')
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re
```


```python
import os
os.chdir('C://Users//leejiwon//Desktop')
```


```python
train = pd.read_csv('open/train.csv', encoding = 'utf-8')
test = pd.read_csv('open/test_x.csv', encoding = 'utf-8')
sample_submission = pd.read_csv('open/sample_submission.csv', encoding = 'utf-8')
```


```python
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>text</th>
      <th>author</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>He was almost choking. There was so much, so m...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>“Your sister asked for it, I suppose?”</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>She was engaged one day as she walked, in per...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>The captain was in the porch, keeping himself ...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>“Have mercy, gentlemen!” odin flung up his han...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>54874</th>
      <td>54874</td>
      <td>“Is that you, Mr. Smith?” odin whispered. “I h...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>54875</th>
      <td>54875</td>
      <td>I told my plan to the captain, and between us ...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>54876</th>
      <td>54876</td>
      <td>"Your sincere well-wisher, friend, and sister...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>54877</th>
      <td>54877</td>
      <td>“Then you wanted me to lend you money?”</td>
      <td>3</td>
    </tr>
    <tr>
      <th>54878</th>
      <td>54878</td>
      <td>It certainly had not occurred to me before, bu...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>54879 rows × 3 columns</p>
</div>




```python
test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>“Not at all. I think she is one of the most ch...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>"No," replied he, with sudden consciousness, "...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>As the lady had stated her intention of scream...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>“And then suddenly in the silence I heard a so...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>His conviction remained unchanged. So far as I...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19612</th>
      <td>19612</td>
      <td>At the end of another day or two, odin growing...</td>
    </tr>
    <tr>
      <th>19613</th>
      <td>19613</td>
      <td>All afternoon we sat together, mostly in silen...</td>
    </tr>
    <tr>
      <th>19614</th>
      <td>19614</td>
      <td>odin, having carried his thanks to odin, proc...</td>
    </tr>
    <tr>
      <th>19615</th>
      <td>19615</td>
      <td>Soon after this, upon odin's leaving the room,...</td>
    </tr>
    <tr>
      <th>19616</th>
      <td>19616</td>
      <td>And all the worse for the doomed man, that the...</td>
    </tr>
  </tbody>
</table>
<p>19617 rows × 2 columns</p>
</div>




```python
sample_submission
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19612</th>
      <td>19612</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19613</th>
      <td>19613</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19614</th>
      <td>19614</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19615</th>
      <td>19615</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19616</th>
      <td>19616</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>19617 rows × 6 columns</p>
</div>




```python
# 부호 제거
def alpha_num(text):
    return re.sub(r'[^A-Za-z0-9 ]', '', text)

train['text']=train['text'].apply(alpha_num)
```


```python
# 부호 사라짐 
train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>text</th>
      <th>author</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>He was almost choking There was so much so muc...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Your sister asked for it I suppose</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>She was engaged one day as she walked in peru...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>The captain was in the porch keeping himself c...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Have mercy gentlemen odin flung up his hands D...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>54874</th>
      <td>54874</td>
      <td>Is that you Mr Smith odin whispered I hardly d...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>54875</th>
      <td>54875</td>
      <td>I told my plan to the captain and between us w...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>54876</th>
      <td>54876</td>
      <td>Your sincere wellwisher friend and sister LUC...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>54877</th>
      <td>54877</td>
      <td>Then you wanted me to lend you money</td>
      <td>3</td>
    </tr>
    <tr>
      <th>54878</th>
      <td>54878</td>
      <td>It certainly had not occurred to me before but...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>54879 rows × 3 columns</p>
</div>




```python
# 불용어 제거
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stopwords:
            final_text.append(i.strip())
    return " ".join(final_text)

# 불용어
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", 
             "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", 
             "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", 
             "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", 
             "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", 
             "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", 
             "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", 
             "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", 
             "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", 
             "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", 
             "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
```


```python
# 전처리 적용
train['text'] = train['text'].str.lower()
test['text'] = test['text'].str.lower()
train['text'] = train['text'].apply(alpha_num).apply(remove_stopwords)
test['text'] = test['text'].apply(alpha_num).apply(remove_stopwords)
```


```python
X_train = np.array([x for x in train['text']])
X_test = np.array([x for x in test['text']])
y_train = np.array([x for x in train['author']])
```


```python
X_train
```




    array(['almost choking much much wanted say strange exclamations came lips pole gazed fixedly bundle notes hand looked odin evident perplexity',
           'sister asked suppose',
           'engaged one day walked perusing janes last letter dwelling passages proved jane not written spirits instead surprised mr odin saw looking odin meeting putting away letter immediately forcing smile said',
           ..., 'sincere wellwisher friend sister lucy odin',
           'wanted lend money', 'certainly not occurred said yes like'],
          dtype='<U1433')




```python
#파라미터 설정
vocab_size = 20000
embedding_dim = 16
max_length = 500
padding_type='post'
#oov_tok = "<OOV>"
```


```python
#tokenizer에 fit
tokenizer = Tokenizer(num_words = vocab_size)#, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
```


```python
#데이터를 sequence로 변환해주고 padding 해줍니다.
train_sequences = tokenizer.texts_to_sequences(X_train)
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(test_sequences, padding=padding_type, maxlen=max_length)
```


```python
#가벼운 NLP모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])
```


```python
# compile model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model summary
print(model.summary())
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 500, 16)           320000    
    _________________________________________________________________
    global_average_pooling1d_1 ( (None, 16)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 24)                408       
    _________________________________________________________________
    dense_3 (Dense)              (None, 5)                 125       
    =================================================================
    Total params: 320,533
    Trainable params: 320,533
    Non-trainable params: 0
    _________________________________________________________________
    None
    


```python
# fit model
num_epochs = 20
history = model.fit(train_padded, y_train, 
                    epochs=num_epochs, verbose=2, 
                    validation_split=0.2)
```

    Train on 43903 samples, validate on 10976 samples
    Epoch 1/20
    43903/43903 - 12s - loss: 1.5643 - acc: 0.2741 - val_loss: 1.5356 - val_acc: 0.2812
    Epoch 2/20
    43903/43903 - 12s - loss: 1.4305 - acc: 0.3861 - val_loss: 1.3251 - val_acc: 0.4389
    Epoch 3/20
    43903/43903 - 12s - loss: 1.2218 - acc: 0.4990 - val_loss: 1.1725 - val_acc: 0.5216
    Epoch 4/20
    43903/43903 - 14s - loss: 1.0976 - acc: 0.5594 - val_loss: 1.0898 - val_acc: 0.5708
    Epoch 5/20
    43903/43903 - 14s - loss: 1.0121 - acc: 0.6009 - val_loss: 1.0270 - val_acc: 0.5970
    Epoch 6/20
    43903/43903 - 15s - loss: 0.9373 - acc: 0.6359 - val_loss: 0.9718 - val_acc: 0.6224
    Epoch 7/20
    43903/43903 - 16s - loss: 0.8683 - acc: 0.6699 - val_loss: 0.9443 - val_acc: 0.6374
    Epoch 8/20
    43903/43903 - 14s - loss: 0.8064 - acc: 0.6973 - val_loss: 0.9283 - val_acc: 0.6346
    Epoch 9/20
    43903/43903 - 15s - loss: 0.7526 - acc: 0.7192 - val_loss: 0.8606 - val_acc: 0.6706
    Epoch 10/20
    43903/43903 - 12s - loss: 0.7082 - acc: 0.7389 - val_loss: 0.8501 - val_acc: 0.6782
    Epoch 11/20
    43903/43903 - 11s - loss: 0.6697 - acc: 0.7554 - val_loss: 0.8147 - val_acc: 0.6981
    Epoch 12/20
    43903/43903 - 13s - loss: 0.6377 - acc: 0.7682 - val_loss: 0.8037 - val_acc: 0.7068
    Epoch 13/20
    43903/43903 - 11s - loss: 0.6067 - acc: 0.7797 - val_loss: 0.7903 - val_acc: 0.7075
    Epoch 14/20
    43903/43903 - 12s - loss: 0.5816 - acc: 0.7896 - val_loss: 0.8145 - val_acc: 0.7008
    Epoch 15/20
    43903/43903 - 12s - loss: 0.5613 - acc: 0.7964 - val_loss: 0.8113 - val_acc: 0.7051
    Epoch 16/20
    43903/43903 - 12s - loss: 0.5390 - acc: 0.8039 - val_loss: 0.8044 - val_acc: 0.7110
    Epoch 17/20
    43903/43903 - 13s - loss: 0.5216 - acc: 0.8106 - val_loss: 0.7806 - val_acc: 0.7180
    Epoch 18/20
    43903/43903 - 14s - loss: 0.5030 - acc: 0.8183 - val_loss: 0.7876 - val_acc: 0.7208
    Epoch 19/20
    43903/43903 - 12s - loss: 0.4872 - acc: 0.8236 - val_loss: 0.7798 - val_acc: 0.7247
    Epoch 20/20
    43903/43903 - 12s - loss: 0.4734 - acc: 0.8271 - val_loss: 0.8042 - val_acc: 0.7160
    


```python
# predict values
pred = model.predict_proba(test_padded)
```


```python
pred
```




    array([[6.77617092e-04, 4.08015907e-01, 9.44074430e-03, 5.79638422e-01,
            2.22728611e-03],
           [8.09224993e-02, 8.60460699e-01, 4.60818317e-03, 9.98526998e-03,
            4.40233573e-02],
           [9.96765852e-01, 2.65704188e-03, 8.35738092e-08, 6.25962437e-09,
            5.77055442e-04],
           ...,
           [3.99159238e-04, 9.99583185e-01, 2.61498767e-10, 1.73376611e-05,
            3.37798241e-07],
           [3.39149643e-04, 9.99643207e-01, 7.85378873e-09, 1.00507605e-05,
            7.67356596e-06],
           [9.99125302e-01, 1.84209421e-04, 5.49841252e-06, 1.17281559e-07,
            6.84881292e-04]], dtype=float32)




```python
# submission
sample_submission[['0','1','2','3','4']] = pred
sample_submission
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.000678</td>
      <td>4.080159e-01</td>
      <td>9.440744e-03</td>
      <td>5.796384e-01</td>
      <td>2.227286e-03</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.080922</td>
      <td>8.604607e-01</td>
      <td>4.608183e-03</td>
      <td>9.985270e-03</td>
      <td>4.402336e-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.996766</td>
      <td>2.657042e-03</td>
      <td>8.357381e-08</td>
      <td>6.259624e-09</td>
      <td>5.770554e-04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.000063</td>
      <td>1.458828e-09</td>
      <td>9.957083e-01</td>
      <td>1.496418e-08</td>
      <td>4.228470e-03</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.949546</td>
      <td>3.233735e-02</td>
      <td>1.294918e-03</td>
      <td>1.451399e-02</td>
      <td>2.307959e-03</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19612</th>
      <td>19612</td>
      <td>0.000001</td>
      <td>9.999988e-01</td>
      <td>1.917964e-20</td>
      <td>9.426437e-10</td>
      <td>1.542850e-14</td>
    </tr>
    <tr>
      <th>19613</th>
      <td>19613</td>
      <td>0.004606</td>
      <td>1.120274e-04</td>
      <td>1.266904e-04</td>
      <td>3.065158e-13</td>
      <td>9.951556e-01</td>
    </tr>
    <tr>
      <th>19614</th>
      <td>19614</td>
      <td>0.000399</td>
      <td>9.995832e-01</td>
      <td>2.614988e-10</td>
      <td>1.733766e-05</td>
      <td>3.377982e-07</td>
    </tr>
    <tr>
      <th>19615</th>
      <td>19615</td>
      <td>0.000339</td>
      <td>9.996432e-01</td>
      <td>7.853789e-09</td>
      <td>1.005076e-05</td>
      <td>7.673566e-06</td>
    </tr>
    <tr>
      <th>19616</th>
      <td>19616</td>
      <td>0.999125</td>
      <td>1.842094e-04</td>
      <td>5.498413e-06</td>
      <td>1.172816e-07</td>
      <td>6.848813e-04</td>
    </tr>
  </tbody>
</table>
<p>19617 rows × 6 columns</p>
</div>




```python
sample_submission.to_csv('submission.csv', index = False, encoding = 'utf-8')
```


```python

```
