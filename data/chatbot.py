import os
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.cache import InMemoryCache
import langchain


# ====== Cấu hình cache (tăng tốc LLM) ======
langchain.llm_cache = InMemoryCache()


# ====== Thông tin hệ thống ======
DATA_PATH = "kien_thuc_giao_duc.txt"
CHROMA_DIR = "data/chroma_db"
OLLAMA_BASE = "http://localhost:11434"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"   # có thể đổi sang qwen2.5:7b hoặc llama3.1:8b cho nhanh hơn


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


# ====== 3) Embedding + Vectorstore ======
print("🔢 Đang tạo embeddings...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE)

vectorstore = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)
vectorstore.persist()
print("💾 Vectorstore đã sẵn sàng.")


# ====== 4) Khởi tạo LLM ======
llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE)


# ====== 5) Prompt rút gọn & tối ưu tốc độ ======
EDU_PROMPT = """
Bạn là "Trợ lý Học vụ CTU" — trợ lý ảo chính thức cho sinh viên, giảng viên và phụ huynh của Trường Đại học Cần Thơ (CTU).

MỤC TIÊU:
- Trả lời chính xác, ngắn gọn, bằng tiếng Việt.
- Nếu câu hỏi liên quan đến thao tác (đăng ký môn, tra cứu điểm, xem lịch thi, lịch học), luôn cung cấp link chính thức.
- Ưu tiên câu trả lời dựa trên dữ liệu RAG (context) → không bịa khi thiếu thông tin.

QUY TẮC:
1. Không được tạo thông tin nếu không chắc chắn. Nếu không có trong dữ liệu RAG, hãy nói: 
   “Dữ liệu này không có trong nguồn hiện có. Bạn có thể xem tại <link chính thức> hoặc liên hệ phòng ban.”
2. Nếu sử dụng dữ liệu từ RAG, phải thêm: [Nguồn: <file hoặc link>].
3. Trả lời theo format:
   - 1–2 câu ngắn tóm tắt
   - Nếu là thao tác: liệt kê bước 1–2–3 + link chính thức
   - Dòng cuối: [Nguồn: ...] + [confidence: XX%]
4. Không xử lý thông tin cá nhân (điểm riêng, MSSV). Chỉ hướng dẫn truy cập hệ thống.
5. Không lan man, không dài dòng, không nói mơ hồ.
6. Luôn ưu tiên các link chính thức:
   - Đăng ký học phần: https://dkmhfe.ctu.edu.vn
   - Tra cứu điểm/lịch thi/lịch học: https://htql.ctu.edu.vn
   - Hỗ trợ kỹ thuật: https://helpdesk.ctu.edu.vn
   - Phòng Đào tạo: pdt@ctu.edu.vn — 0292 383 1156
   - Phòng CTSV: pctsv@ctu.edu.vn — 0292 387 2177

OUTPUT:
- Trả lời ngắn gọn (3–6 câu), có nguồn, có confidence.
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=EDU_PROMPT + "\n\nThông tin tham khảo:\n{context}\n\nCâu hỏi: {question}\nTrả lời ngắn gọn:"
)


# ====== 6) Vòng lặp giao tiếp ======
print("\n🎓 Chatbot CTU đã sẵn sàng! (gõ 'exit' để thoát)\n")

while True:
    q = input("👩‍🎓 Bạn: ").strip()
    if q.lower() == "exit":
        print("👋 Tạm biệt! Chúc bạn một ngày tốt lành!")
        break

    try:
        # Lấy context nhanh (tối ưu k=2)
        results = vectorstore.similarity_search(q, k=2)
        context = "\n\n".join([doc.page_content for doc in results])

        # Tạo prompt cuối
        final_prompt = prompt.format(context=context, question=q)

        # Gọi LLM với temperature thấp (trả lời nhanh & ít suy nghĩ)
        print("🤖 Trợ lý CTU: ", end="", flush=True)

        for chunk in llm.stream(final_prompt):
            print(chunk, end="", flush=True)

        print("\n")  # xuống dòng sau khi stream xong

    except Exception as e:
        print(f"⚠️ Lỗi: {e}\n")
