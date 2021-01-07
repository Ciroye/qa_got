from haystack import Finder
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.retriever.sparse import TfidfRetriever
import streamlit as st

#max_width_str = f"max-width: 1600px;"
#st.markdown(f"""<style> .reportview-container .main .block-container{{{max_width_str}}}</style>""", True)

st.markdown("<center> <h1> 📜 Questions And Answering Game of thrones </h1> </center>", True)

@st.cache
def read_wikipedia():
    document_store = InMemoryDocumentStore()
    doc_dir = "data/article_txt_got"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)
    dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
    document_store.write_documents(dicts)
    return document_store

@st.cache
def retriever():
    document_store = read_wikipedia()
    retriever = TfidfRetriever(document_store=document_store)
    return retriever

question = st.text_input('Input your question here:')

if st.button('Ask'):
    with st.spinner('Reading all the books and  watching all the seasons'):
        retriever = retriever()
        reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
        finder = Finder(reader, retriever)
        prediction = finder.get_answers(question=question, top_k_retriever=10, top_k_reader=1)
        st.info(prediction['answers'][0]['answer'])


donate = """<form action="https://www.paypal.com/cgi-bin/webscr" method="post" target="_top">
<input type="hidden" name="cmd" value="_donations" />
<input type="hidden" name="business" value="DHGY7GMDTS6TA" />
<input type="hidden" name="currency_code" value="USD" />
<input style="float:right; margin-right: 80px width: 180px" type="image" src="https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif" border="0" name="submit" title="PayPal - The safer, easier way to pay online!" alt="Donate with PayPal button" />
</form>"""
image = """<img src="https://www.itl.cat/pngfile/big/2-25471_ultra-hd-game-of-thrones.jpg"
alt="Game of thrones"
style="width: 100%; height: 300px;">"""
st.markdown(image, True)
st.markdown(donate, True)

