# Install all of it
#!pip install python-dotenv
#!pip install -q -U google-generativeai
#!pip install -U langchain-community
#!pip install faiss-cpu
#!pip install langchain_google_genai
#!pip install gradio
# import all library, framework and load all model we need
from joblib import load
import gradio as gr
import re
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models,transforms
import pickle
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
nltk.download('stopwords')
# load all model and API
device = 0 if torch.cuda.is_available() else -1 # check if GPU available using GPU
translator_eng_id = pipeline('translation', model='Helsinki-NLP/opus-mt-en-id', device = device)# eng-indo
translator_id_eng = pipeline('translation', model='Helsinki-NLP/opus-mt-id-en', device = device)# indo-eng
summarizer = pipeline('summarization', model='facebook/bart-large-cnn', device = device) # summarize the news
ner = pipeline('ner',tokenizer = 'dbmdz/bert-large-cased-finetuned-conll03-english', model='dbmdz/bert-large-cased-finetuned-conll03-english', device = device) # Analyze NER on news
lang_detector = pipeline('text-classification', model = 'papluca/xlm-roberta-base-language-detection', device = device) #detect language
news_encoder = load('news_encoder.pkl') #encoder for the news (3 class)
news_classifier = load("news_classifier.pkl") #classify news into 3 category (Politic, Technology, Science)
emotion_text_encoder = load("emotion_encoder.pkl") # Encoder for Emotion detection(text version)
emotion_classifier = load("emotion_model_classifier.pkl") #Detect emotion user based on their comment about news
with open('class_names_face_recognition.pkl', 'rb') as f: # import and using it as encoder for emotion detection(image version)
    emotion_face_class = pickle.load(f)
face_emotion = models.resnet18(pretrained=False) # Detect user emotion based on their expression (selfie)
face_emotion.fc = nn.Linear(face_emotion.fc.in_features, 5)
#face_emotion = face_emotion.load_state_dict(torch.load('Best_emotion_face_model.pth'),map_location=torch.device('cpu'))
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') # move into GPU if available, so model will using GPU
state_dict = torch.load('Best_emotion_face_model.pth', map_location=device)
face_emotion.load_state_dict(state_dict)
face_emotion.to(device)
face_emotion.eval()
# using API Gemini for chatbot
GOOGLE_API_KEY = "YOUR API"#put your API here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Support Function
#detect language
def detect_language(text):
  result = lang_detector(text)
  return result[0]['label']
#translate to english if user language not english
def translate_to_english(text,source_lang):
  if source_lang.lower() == 'indonesia':
    return translator_id_eng(text)[0]['translation_text']
  else:
    return text
def translate_to_original(text,target_lang):
  if target_lang.lower() == 'indonesia':
    return translator_eng_id(text)[0]['translation_text']
  else:
    return text
# upload reaction
def process_image_reaction(image_path):
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = image.astype(np.float32)/255.0
  image = np.transpose(image, (2,0,1))
  image = torch.tensor(image).unsqueeze(0).to(device)
  #predict
  with torch.no_grad():
    outputs = face_emotion(image)
    _,predicted = torch.max(outputs,1)
  return emotion_face_class[predicted.item()]
#make template  for cleaning
def cleaning_text(text, is_lower_case = False):
  text = text.lower()
  pattern = r'[^a-zA-z0-9\s]'
  text = re.sub(pattern,'',text)
  tokenizer = ToktokTokenizer()
  stopword = stopwords.words('english')
  tokens = tokenizer.tokenize(text)
  tokens = [token.strip() for token in tokens]
  if is_lower_case:
    filtered_tokens = [token for token in tokens if token not in stopword]
  else:
    filtered_tokens = [token for token in tokens if token.lower() not in stopword]
  filtered_text =' '.join(filtered_tokens)
  st = LancasterStemmer()
  text = ' '.join([st.stem(word) for word in filtered_text.split()])
  return text

#Step 1 to classify, summary and get NER from news
def extract_article(url):
   model = genai.GenerativeModel('gemini-1.5-flash')
   extract = model.generate_content(f"Extract the content from this news article: {url}, make sure extract all news part. Extract based on the news language used, if the news using Indonesia language just extract with Indonesian language, and so on if the news using english language. Make sure minimum is 100 words and maximum is 200 words").text
   return extract
def process_news_pipeline(news_text, user_language):
    news_text = extract_article(news_text)
    #detect the  language
    detected_lang = detect_language(news_text)
#translate news if not in english
    if detected_lang.lower() != 'english':
        news_text = translate_to_english(news_text,detected_lang)
    #analyze the news
    #classify the news category
    summary = summarizer(news_text, min_length = 25, do_sample = False)[0]['summary_text']
    clean_text = cleaning_text(summary)
    classification = news_classifier.predict([clean_text])[0]
    classification = news_encoder.inverse_transform([classification])[0]
    ner_result = ner(news_text)
    ner_filter = [{"entity":result['entity'],'word':result['word']} for result in ner_result]
    ner_filter = ner_filter[:5]
    #translate result into original language if not english
    if user_language.lower() != 'english':
        classification = translate_to_original(classification, user_language)
        summary = translate_to_original(summary, user_language)
    return classification, summary, ner_filter
def bring_together_all_of_it(news_text, user_language):
    news_text = extract_article(news_text)
    classification, summary, ner_results = process_news_pipeline(news_text,user_language)
    ner_results = "\n".join([f"Entity: {item['entity']}, Word: {item['word']}" for item in ner_results])
    return classification, summary, ner_results, news_text

#Step 2 for predict user emotion based on text or image
def process_user_response(response_type, response_content, user_language):
    if response_type == "Text" and response_content:
      sentiment = emotion_text_encoder.inverse_transform([emotion_classifier.predict([response_content])[0]])[0] #predict output using sentiment analysis model
      if user_language != 'English':
        translate = translate_to_english(response_content,user_language)
        sentiment = emotion_text_encoder.inverse_transform([emotion_classifier.predict([translate])[0]])[0] #predict output using sentiment analysis model
        sentiment = translate_to_original(sentiment,user_language)
    elif response_type == "Selfie" and response_content:
      sentiment = process_image_reaction(response_content) # predict output using image classification model
      if user_language != 'English':
        sentiment = translate_to_original(sentiment,user_language)
    else:
      return "Please provide valid input."
    return f"Your Expression: {sentiment}"
 
#Step 3, make chatbot for discussion about the news
#load model and memory, to save context while discuss with chatbot
model = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=model,
    verbose=False,
    memory=memory
)
def chatbot_interaction(user_message, history, news, user_reaction, user_language):
  logo = "https://i.ibb.co.com/rwft0JQ/output-removebg-preview-1.png"
  if history is None:
    history = []
  # first response based on bot response to know the context and save it into history
  if len(history) == 0:
      initial_response = f"Hi genius! wanna discuss something?"
      memory.chat_memory.add_ai_message(initial_response)
      memory.chat_memory.add_user_message(f"Learn and become good presenter based on this news {news}, after that you will discuss with user about this news and make sure using {user_language} language and their ekspression : {user_reaction}")
      history.append([f"👤 {user_message}", f"<img src='{logo}' style='width:30px; height:30px;'> { initial_response}"])
      return history, ""
  memory.chat_memory.add_user_message(user_message)
  gpt_response = conversation.predict(input=user_message)
  memory.chat_memory.add_ai_message(gpt_response)
  history.append([f"👤 {user_message}",f"<img src='{logo}' style='width:30px; height:30px;'> {gpt_response}"] )
  return history, ""

#Building UI from Gradio
#Step 1
with gr.Blocks() as news_analysis_interface:
  news_text = gr.Textbox(label="Put the news link")
  user_language = gr.Radio(choices=["English", "Indonesia"], label="User Language")
  submit_analysis = gr.Button("Analyze News")
  summary_result = gr.Textbox(label="News Classification", elem_id="summary_result_box")
  classification_result = gr.Textbox(label="News Summary")
  ner_result = gr.Textbox(label="Named Entities")
  news_state = gr.Textbox(label="News")
  submit_analysis.click(
      fn=bring_together_all_of_it,
      inputs=[news_text, user_language],
      outputs=[summary_result, classification_result, ner_result, news_state]
  )

#step 2
with gr.Blocks() as reaction_interface:
  response_type = gr.Radio(["Text", "Selfie"], label="Response Type")
  text_input = gr.Textbox(label="Enter Text (only understand english comment)", visible=False)
  image_input = gr.Image(type="filepath", label="Upload Image", visible=False)
  reaction_result = gr.Textbox(label="Detected Reaction", interactive=False)
  def update_visibility(selected_response):
      if selected_response == "Text":
          return gr.update(visible=True), gr.update(visible=False)
      elif selected_response == "Selfie":
          return gr.update(visible=False), gr.update(visible=True)
      else:
          return gr.update(visible=False), gr.update(visible=False)
  response_type.change(
      fn=update_visibility,
      inputs=response_type,
      outputs=[text_input, image_input]
  )
  submit_button = gr.Button("Submit Reaction")
  def submit_click(response_type_value, text_value, image_value,language_value):
    if response_type_value == "Text" and text_value:
      processed_reaction = process_user_response(response_type_value, text_value,language_value)
      return processed_reaction
    elif response_type_value == "Selfie" and image_value:
      processed_reaction = process_user_response(response_type_value, image_value,language_value)
      return processed_reaction
    else:
      return "Please provide valid input."
  submit_button.click(
      fn = submit_click,
      inputs=[response_type, text_input, image_input, user_language],
      outputs=[reaction_result]
  )

#step 3
with gr.Blocks() as chatbot_interface:
  chatbot_chatbox = gr.Chatbot(label="Chatbot Interaction")
  chatbot_input = gr.Textbox(label="Your Message", placeholder="Type your message here...")
  chatbot_submit_button = gr.Button("Send")
  chatbot_history = gr.State(value=[])
  chatbot_submit_button.click(
      fn=chatbot_interaction,
      inputs=[chatbot_input,chatbot_history,news_state,reaction_result,user_language],
      outputs=[chatbot_chatbox,chatbot_input]
  )
  chatbot_input.submit(fn=chatbot_interaction,
                        inputs=[chatbot_input,chatbot_history,news_state,reaction_result,user_language],
      outputs=[chatbot_chatbox,chatbot_input]
  )

#Bring it all and launch
with gr.Blocks(theme=gr.themes.Default(primary_hue='gray', secondary_hue='zinc')) as full_app:
  with gr.Column():
    # Membuat gambar terpusat di atas
    gr.HTML("""
      <div style="text-align: center;">
          <img src="https://i.ibb.co.com/rwft0JQ/output-removebg-preview-1.png" alt="Robot Logo" style="width:150px; display: block; margin: 0 auto;">
      </div>
      """)
    # Menambahkan judul
    gr.Markdown("<h1 style='text-align: center;'>Your Discussion Friend</h1>")
  #gr.Markdown("# Your Discussion Friend")
  gr.Markdown("**Step 1**: Analyze the News")
  news_analysis_interface.render()
  gr.Markdown("**Step 2**: Choose Your Reaction")
  reaction_interface.render()
  gr.Markdown("**Step 3**: Interact with Chatbot")
  chatbot_interface.render()
full_app.launch()
