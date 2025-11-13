import os
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.cache import InMemoryCache
import langchain

# ====== Cấu hình cache để tăng tốc ======
langchain.llm_cache = InMemoryCache()

# ====== Cấu hình chatbot ======
DATA_PATH = "kien_thuc_giao_duc.txt"
CHROMA_DIR = "data/chroma_db"
OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma2:9b"

# ====== 1) Load dữ liệu ======
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {DATA_PATH}")

print("📘 Đang tải dữ liệu...")
loader = TextLoader(DATA_PATH, encoding="utf-8")
documents = loader.load()

# ====== 2) Chia nhỏ văn bản ======
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"✅ Đã chia thành {len(chunks)} đoạn.")

# ====== 3) Tạo embeddings + Chroma vectorstore ======
print("🔢 Đang tạo embeddings...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE)

vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
vectorstore.persist()
print("💾 Vectorstore đã sẵn sàng.")

# ====== 4) Khởi tạo LLM ======
llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE)

# ====== 5) Prompt Template ======
EDU_PROMPT = (
    "Bạn là trợ lý ảo của Trường Đại học Cần Thơ (CTU), chuyên hỗ trợ sinh viên về học vụ và hành chính. "
    "Chỉ trả lời các câu hỏi liên quan đến sinh viên, tuyển sinh, học vụ, học phí, đăng ký học phần, quy định, "
    "và thông tin liên hệ trong trường. "
    "Nếu câu hỏi nằm ngoài các lĩnh vực này (ví dụ: thời sự, lập trình, thời tiết, giải trí, chính trị...), "
    "hãy trả lời: 'Xin lỗi, tôi chỉ hỗ trợ thông tin liên quan đến học tập và sinh viên Trường Đại học Cần Thơ.' "
    "\n\nDựa trên ngữ cảnh được cung cấp, hãy trả lời ngắn gọn, chính xác và bằng tiếng Việt thân thiện."
)
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=EDU_PROMPT + "\n\nNgữ cảnh:\n{context}\n\nCâu hỏi: {question}\nTrả lời:"
)

# ====== 6) Vòng lặp hỏi đáp ======
print("\n🎓 Chatbot sinh viên CTU đã sẵn sàng! (gõ 'exit' để thoát)\n")

while True:
    q = input("👩‍🎓 Bạn: ").strip()
    if q.lower() == "exit":
        print("👋 Tạm biệt! Tôi rất vui vì đã hỗ trợ bạn.")
        break

    try:
        # 6.1 Tìm các đoạn liên quan
        top_k = 3
        results = vectorstore.similarity_search(q, k=top_k)
        context = "\n\n".join([doc.page_content for doc in results]) if results else ""

        # Nếu không có dữ liệu phù hợp
        if not context:
            print("🤖 Trợ lý CTU: Tôi chưa có dữ liệu về nội dung này.\n")
            continue

        # 6.2 Ghép prompt
        final_prompt = prompt.format(context=context, question=q)

        # 6.3 Gọi LLM
        answer = llm.invoke(final_prompt)
        print(f"🤖 Trợ lý CTU: {answer}\n")

    except Exception as e:
        print(f"⚠️ Lỗi khi xử lý câu hỏi: {e}\n")
