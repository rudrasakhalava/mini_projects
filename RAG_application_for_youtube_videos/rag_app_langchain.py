from dotenv import load_dotenv
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load Environment Variables
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key,
    temperature=0.2,
)

# Get YouTube Transcript
video_id = "twrHreKqW6w"

try:
    ytt_api = YouTubeTranscriptApi()

    transcript_list = ytt_api.fetch(
        video_id,
        languages=["hi"]        # change to ["en"] if needed
    )

    transcript = " ".join(
        chunk.text for chunk in transcript_list
    )

except TranscriptsDisabled:
    print("No captions available.")
    exit()

# Split Transcript
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

chunks = splitter.create_documents([transcript])

# Embeddings + FAISS
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = FAISS.from_documents(
    chunks,
    embeddings,
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

# Prompt
prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer ONLY using the transcript context below.

If the answer is not present in the transcript,
reply with:
"I don't know based on the transcript."

Transcript:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)

# Helper Function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

# Ask Questions
while True:

    query = input("\nAsk Question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    answer = main_chain.invoke(query)

    print("\nAnswer:\n")
    print(answer)
