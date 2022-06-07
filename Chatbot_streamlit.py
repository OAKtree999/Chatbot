import streamlit as st 
from streamlit_chat import message 
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

df = pd.read_csv('wellness_dataset.csv', sep=',', encoding = 'UTF-8') # csv 파일 읽어오기
df.head() # csv 파일 불러오기

# https://docs.streamlit.io/library/advanced-features/caching
# 웹에서 데이터를 로딩할 때 효율적으로 앱에 stay할 수 있는 캐쉬 메커니즘
# 데이터베이스 연결이 캐쉬에 저장되어진다. 매번 다음에 cached_model을 불러오고, 이미 형성되고 연결된 object는 자동으로 재활용한다.
# SentenceTransformer 모듈을 넣고 필요할 때마다 가져오는 것으로 생각됨.
@st.cache(allow_output_mutation=True) 
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask') # https://github.com/jhgan00/ko-sentence-transformers
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('wellness_dataset.csv', sep=',', encoding = 'UTF-8')
    df['embedding'] = df['embedding'].apply(json.loads) # embedding column을 json으로 변경
    return df

model = cached_model()
df = get_dataset()

st.header('심리 상담 챗봇') # streamlit 해더
st.markdown('**상담사 온안국**') 

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("상담하고 싶은 내용을 적으세요 : ","", key="input")
    return input_text 

user_input = get_text()


if user_input:

    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer['챗봇'])

    
for i in range(len(st.session_state['past'])-1,-1,-1):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')

# streamlit run Chatbot_streamlit.py