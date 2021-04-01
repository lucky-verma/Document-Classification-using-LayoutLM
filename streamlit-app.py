import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from transformers import LayoutLMForSequenceClassification, LayoutLMTokenizer
import torch
import requests
from torch.utils.data import Dataset, DataLoader
import pytesseract
from datasets import Features, Sequence, ClassLabel, Value, Array2D
import numpy as np
import streamlit as st
from datasets import Dataset
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# Legacy method imports

def normalize_box(box, width, height):
     return [
         int(1000 * (box[0] / width)),
         int(1000 * (box[1] / height)),
         int(1000 * (box[2] / width)),
         int(1000 * (box[3] / height)),
     ]

def apply_ocr(example):
        # get the image
        image = Image.open(example['image_path'])

        width, height = image.size
        
        # apply ocr to the image 
        ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
        float_cols = ocr_df.select_dtypes('float').columns
        ocr_df = ocr_df.dropna().reset_index(drop=True)
        ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
        ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
        ocr_df = ocr_df.dropna().reset_index(drop=True)

        # get the words and actual (unnormalized) bounding boxes
        #words = [word for word in ocr_df.text if str(word) != 'nan'])
        words = list(ocr_df.text)
        words = [str(w) for w in words]
        coordinates = ocr_df[['left', 'top', 'width', 'height']]
        actual_boxes = []
        for idx, row in coordinates.iterrows():
            x, y, w, h = tuple(row) # the row comes in (left, top, width, height) format
            actual_box = [x, y, x+w, y+h] # we turn it into (left, top, left+width, top+height) to get the actual box 
            actual_boxes.append(actual_box)
        
        # normalize the bounding boxes
        boxes = []
        for box in actual_boxes:
            boxes.append(normalize_box(box, width, height))
        
        # add as extra columns 
        assert len(words) == len(boxes)
        example['words'] = words
        example['bbox'] = boxes
        return example

tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

def encode_example(example, max_seq_length=512, pad_token_box=[0, 0, 0, 0]):
  words = example['words']
  normalized_word_boxes = example['bbox']

  assert len(words) == len(normalized_word_boxes)

  token_boxes = []
  for word, box in zip(words, normalized_word_boxes):
      word_tokens = tokenizer.tokenize(word)
      token_boxes.extend([box] * len(word_tokens))
  
  # Truncation of token_boxes
  special_tokens_count = 2 
  if len(token_boxes) > max_seq_length - special_tokens_count:
      token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
  
  # add bounding boxes of cls + sep tokens
  token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
  
  encoding = tokenizer(' '.join(words), padding='max_length', truncation=True)
  # Padding of token_boxes up the bounding boxes to the sequence length.
  input_ids = tokenizer(' '.join(words), truncation=True)["input_ids"]
  padding_length = max_seq_length - len(input_ids)
  token_boxes += [pad_token_box] * padding_length
  encoding['bbox'] = token_boxes

  assert len(encoding['input_ids']) == max_seq_length
  assert len(encoding['attention_mask']) == max_seq_length
  assert len(encoding['token_type_ids']) == max_seq_length
  assert len(encoding['bbox']) == max_seq_length

  return encoding

# we need to define the features ourselves as the bbox of LayoutLM are an extra feature
features = Features({
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'image_path': Value(dtype='string'),
    'words': Sequence(feature=Value(dtype='string')),
})

classes = ["bill", "invoice", "others", "Purchase_Order", "remittance"]


# Model Loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@st.cache(allow_output_mutation=True)
def load_model():
    url = "https://vast-ml-models.s3-ap-southeast-2.amazonaws.com/Document-Classification-5-labels-final.bin"
    r = requests.get(url, allow_redirects=True)
    open('saved_model/pytorch_model.bin', 'wb').write(r.content)
    model = LayoutLMForSequenceClassification.from_pretrained("saved_model")
    return model

load_model().to(device)

# Data processing

st.title('VAST: Document Classifier')
st.header('Upload any document image')



image = st.file_uploader('Upload here', type=['jpg', 'png', 'jpeg', 'webp'])

if image is None:
    st.write("### Please upload your Invoice IMAGE")
else:
    im = Image.open(image)
    rgb_im = im.convert('RGB')
    rgb_im.save('test_data/audacious.jpg')
    os.getcwd()
    test_data = pd.DataFrame.from_dict({'image_path': ['test_data/audacious.jpg']})
    st.image(image, caption='your_doc', use_column_width=True)
    if st.button("Process"):
        st.spinner()
        with st.spinner(text='In progress'):
            test_dataset = Dataset.from_pandas(test_data)
            updated_test_dataset = test_dataset.map(apply_ocr)
            st.success('OCR Done')
            encoded_test_dataset = updated_test_dataset.map(lambda example: encode_example(example), 
                                      features=features)
            encoded_test_dataset.set_format(type='torch', columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids'])
            test_dataloader = torch.utils.data.DataLoader(encoded_test_dataset, batch_size=1, shuffle=True)
            test_batch = next(iter(test_dataloader))
            st.success('Encoding Data Done')
            input_ids = test_batch["input_ids"].to(device)
            bbox = test_batch["bbox"].to(device)
            attention_mask = test_batch["attention_mask"].to(device)
            token_type_ids = test_batch["token_type_ids"].to(device)

            # forward pass
            outputs = load_model()(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, 
                            token_type_ids=token_type_ids)
            
            classification_logits = outputs.logits
            classification_results = torch.softmax(classification_logits, dim=1).tolist()[0]
            
            # Show JSON output
            thisdict ={}
            for i in range(len(classes)):
                thisdict[classes[i]] = str(int(round(classification_results[i] * 100))) + "%"
            st.json(thisdict)
            
            # Show a Plotly Graph
            res_list = []
            res_dict ={"Type of Document":["bill", "invoice", "others", "Purchase_Order", "remittance"],
                      "Prediction Percent": res_list}
            for i in range(len(classes)):
                res_list.append(classification_results[i] * 100)
                res_dict[classes[i]] = int(round(classification_results[i] * 100))
            total_dataframe = pd.DataFrame(res_dict)
            state_total_graph = px.bar(
            total_dataframe, 
            x='Type of Document',
            y='Prediction Percent',
            labels={'YOYO': 'Prediction Percent' }, color='Type of Document')
            st.plotly_chart(state_total_graph)
            

        st.success('Done')
        st.balloons()
    
