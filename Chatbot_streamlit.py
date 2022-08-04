import streamlit as st 
from streamlit_chat import message 
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

df = pd.read_csv('wellness_dataset.csv', sep=',', encoding = 'UTF-8') # csv 파일 읽어오기
df.head() # csv 파일 불러오기

@st.cache(allow_output_mutation=True) 
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask') 
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('wellness_dataset.csv', sep=',', encoding = 'UTF-8')
    df['embedding'] = df['embedding'].apply(json.loads) # embedding column을 json으로 변경
    return df

model = cached_model()
df = get_dataset()

st.header('심리 상담 챗봇 테스트 중') # streamlit 해더
st.markdown('**상담사 타오른다봇**') # 

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


with st.form('form', clear_on_submit=True):
    user_input = st.text_input('하고싶은 말을 적어보세요: ', '', key="input")
    submitted = st.form_submit_button('상담 받기')

if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer['챗봇'])

for i in range(len(st.session_state['past'])-1,-1,-1):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')

# streamlit run chatbot_test.py
