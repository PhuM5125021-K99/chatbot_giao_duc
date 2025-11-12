# chatbot.py — phiên bản dùng similarity_search (không cần RetrievalQA import)
import os
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


# ====== Cấu hình ======
DATA_PATH = "kien_thuc_giao_duc.txt"  # đường dẫn file dữ liệu văn bản
CHROMA_DIR = "data/chroma_db"
OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"   # nếu Ollama của bạn có model embedding khác, đổi cho phù hợp
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

# ====== 3) Tạo embeddings + Chroma vectorstore (nếu chưa có) ======
# Nếu bạn muốn tái sử dụng DB đã tồn tại (để không phải tạo lại mỗi lần),
# có thể kiểm tra CHROMA_DIR tồn tại rồi load thay vì rebuild.
print("🔢 Tạo embeddings và lưu vào Chroma...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE)

# Tạo/ghi Chroma DB
vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
vectorstore.persist()
print("💾 Vectorstore đã sẵn sàng.")

# ====== 4) Khởi tạo LLM (Ollama) ======
llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE)

# ====== 5) Prompt template ======
EDU_PROMPT = (
    "Bạn là một trợ lý AI chuyên về giáo dục, thân thiện và trả lời bằng tiếng Việt. "
    "Dựa trên ngữ cảnh được cung cấp, hãy trả lời câu hỏi một cách rõ ràng và chi tiết. "
    "Nếu thông tin không có trong ngữ cảnh, hãy nói 'Tôi chưa có dữ liệu về nội dung này.'"
)
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=EDU_PROMPT + "\n\nNgữ cảnh:\n{context}\n\nCâu hỏi: {question}\nTrả lời:"
)

# ====== 6) Vòng lặp hỏi đáp: tìm đoạn liên quan + sinh câu trả lời ======
print("\n🎓 Chatbot giáo dục sẵn sàng! (gõ 'exit' để thoát)\n")

while True:
    q = input("👩‍🎓 Bạn: ").strip()
    if q.lower() == "exit":
        print("👋 Tạm biệt! Chúc bạn học tốt.")
        break

    try:
        # 6.1 Tìm top-k đoạn liên quan (similarity search)
        top_k = 3
        results = vectorstore.similarity_search(q, k=top_k)  # trả về list Document
        context = "\n\n".join([doc.page_content for doc in results]) if results else ""

        if not context:
            # Nếu không tìm thấy đoạn nào, cho thông báo ngắn rồi vẫn gọi LLM (hoặc bỏ qua)
            print("🤖 Trợ lý: Tôi chưa có dữ liệu về nội dung này.\n")
            continue

        # 6.2 Ghép prompt với context
        final_prompt = prompt.format(context=context, question=q)

        # 6.3 Gọi LLM để sinh câu trả lời
        answer = llm.invoke(final_prompt)

        print(f"🤖 Trợ lý: {answer}\n")

    except Exception as e:
        print(f"⚠️ Lỗi khi xử lý câu hỏi: {e}\n")
