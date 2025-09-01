# main.py - Исправленный и улучшенный код для Cloud Run
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import pinecone
from datetime import datetime
import traceback # Для детального логирования исключений

# --- Настройка логирования ---
# В production используйте уровень WARNING или ERROR
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Глобальные переменные для хранения инициализированных клиентов ---
embedding_model = None
gemini_model = None
pinecone_index = None

# --- Конфигурация ---
PROJECT_ID = os.environ.get("PROJECT_ID", "ai-project-26082025")
REGION = os.environ.get("REGION", "us-central1")
# --- Конфигурация Pinecone ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "med-index") # Убедитесь, что имя правильное


# --- Lifespan handler для FastAPI ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, gemini_model, pinecone_index
    # --- Инициализация при запуске ---
    logger.info("🚀 Начало инициализации приложения Medical RAG API...")

    try:
        # 1. Инициализация Vertex AI
        # В Cloud Run аутентификация обычно происходит автоматически через сервисный аккаунт
        # Не нужно явно указывать файл ключа, если он прикреплен к сервису
        logger.info(f"🔧 Инициализация Vertex AI: project={PROJECT_ID}, location={REGION}")
        vertexai.init(project=PROJECT_ID, location=REGION)
        logger.info("✅ Vertex AI инициализирована.")

        # 2. Инициализация моделей Vertex AI
        logger.info("🧠 Загрузка модели эмбеддингов text-embedding-005...")
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        logger.info("✅ Модель эмбеддингов загружена.")

        logger.info("🧠 Загрузка модели генерации gemini-2.5-pro...")
        gemini_model = GenerativeModel("gemini-2.5-pro") # Убедитесь, что модель доступна
        logger.info("✅ Модель генерации загружена.")

        # 3. Инициализация Pinecone
        logger.info("🔗 Инициализация Pinecone...")
        if not PINECONE_API_KEY:
            raise ValueError("❌ Переменная окружения PINECONE_API_KEY не установлена!")
        if not PINECONE_INDEX_NAME:
            raise ValueError("❌ Переменная окружения PINECONE_INDEX_NAME не установлена!")

        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)

        # Проверка подключения и получения статистики
        index_stats = pinecone_index.describe_index_stats()
        logger.info(f"✅ Pinecone инициализирован. Индекс '{PINECONE_INDEX_NAME}' содержит {index_stats.get('total_vector_count', 0)} векторов.")
        logger.info(f"   Размерность векторов: {index_stats.get('dimension', 'N/A')}")

    except Exception as e:
        logger.error(f"💥 Критическая ошибка инициализации: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        # В production можно бросить исключение, чтобы сервис не стартовал
        # raise
        # Но для диагностики пусть запустится, но с ошибками
        embedding_model = None
        gemini_model = None
        pinecone_index = None

    logger.info("🟢 Приложение готово к обработке запросов.")
    yield  # Приложение работает
    # --- Очистка при завершении ---
    logger.info("🛑 Завершение работы приложения...")

# --- Создание приложения FastAPI с lifespan handler ---
app = FastAPI(
    lifespan=lifespan,
    title="Medical RAG API",
    version="2.1.0-Pinecone-Fixed",
    description="Исправленная версия API для медицинского ассистента с RAG"
)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Для разработки. В production укажите конкретные origins.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Модели данных ---
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]

# --- Эндпоинты API ---
@app.get("/")
async def home():
    """Корневой эндпоинт для проверки состояния сервиса."""
    try:
        vector_count = 0
        if pinecone_index:
            try:
                stats = pinecone_index.describe_index_stats()
                vector_count = stats.get('total_vector_count', 'N/A')
            except Exception as e:
                logger.warning(f"Не удалось получить статистику Pinecone в /: {e}")
                vector_count = f"Ошибка: {e}"
        else:
            vector_count = "Не инициализирован"

        return {
            "status": "ok",
            "message": "Medical RAG API Server on Google Cloud Run (Pinecone + Vertex AI)",
            "version": "2.1.0-Pinecone-Fixed",
            "pinecone_vectors": vector_count,
            "models_initialized": {
                "embedding_model": embedding_model is not None,
                "gemini_model": gemini_model is not None,
                "pinecone_index": pinecone_index is not None
            }
        }
    except Exception as e:
        logger.error(f"Ошибка в корневом эндпоинте: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Проверка состояния сервиса."""
    try:
        # Проверяем Pinecone
        pinecone_status = "uninitialized"
        vector_count = 0
        dimension = 'N/A'
        if pinecone_index is not None:
            try:
                stats = pinecone_index.describe_index_stats()
                pinecone_status = "healthy"
                vector_count = stats.get('total_vector_count', 0)
                dimension = stats.get('dimension', 'N/A')
            except Exception as e:
                pinecone_status = "unhealthy"
                logger.error(f"Health check - Pinecone error: {e}")

        # Проверяем модели
        embedding_model_status = "healthy" if embedding_model is not None else "uninitialized"
        gemini_model_status = "healthy" if gemini_model is not None else "uninitialized"

        overall_status = "healthy"
        if pinecone_status != "healthy" or embedding_model_status != "healthy" or gemini_model_status != "healthy":
            overall_status = "degraded"
            if pinecone_status == "uninitialized" and embedding_model_status == "uninitialized" and gemini_model_status == "uninitialized":
                overall_status = "uninitialized"

        return {
            "status": overall_status,
            "components": {
                "pinecone": {
                    "status": pinecone_status,
                    "vector_count": vector_count,
                    "dimension": dimension
                },
                "embedding_model": {
                    "status": embedding_model_status
                },
                "gemini_model": {
                    "status": gemini_model_status
                }
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        logger.error(f"Ошибка в /health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Эндпоинт для ответа на медицинские вопросы с использованием RAG.
    """
    question = request.question.strip()
    logger.info(f"📥 Получен вопрос: {question}")

    # 1. Валидация входных данных
    if not question:
        logger.warning("Получен пустой вопрос")
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")
    if len(question) > 1000:
        logger.warning(f"Получен слишком длинный вопрос ({len(question)} символов)")
        raise HTTPException(status_code=400, detail="Вопрос слишком длинный (максимум 1000 символов)")

    # 2. Проверка инициализации компонентов
    if not embedding_model:
        logger.error("Модель эмбеддингов не инициализирована")
        raise HTTPException(status_code=500, detail="Сервис не готов: модель эмбеддингов не загружена")
    if not gemini_model:
        logger.error("Модель генерации не инициализирована")
        raise HTTPException(status_code=500, detail="Сервис не готов: модель генерации не загружена")
    if not pinecone_index:
        logger.error("Индекс Pinecone не инициализирован")
        raise HTTPException(status_code=500, detail="Сервис не готов: база знаний недоступна")

    try:
        # 3. Создание эмбеддинга для вопроса
        logger.debug("🧠 Создание эмбеддинга для вопроса...")
        embedding_response = embedding_model.get_embeddings([question])
        question_embedding = embedding_response[0].values
        logger.debug(f"   Эмбеддинг создан (размерность: {len(question_embedding)})")

        # 4. Поиск похожих документов в Pinecone
        logger.debug("🔍 Поиск в Pinecone...")
        search_results = pinecone_index.query(
            vector=question_embedding,
            top_k=3,
            include_metadata=True
        )
        logger.debug(f"   Поиск завершен. Найдено {len(search_results.matches)} результатов.")

        # 5. Обработка результатов поиска
        contexts = []
        sources = []
        if search_results.matches:
            for match in search_results.matches:
                metadata = match.metadata or {}
                # Адаптируйте ключи под структуру ваших метаданных в Pinecone
                # Предположим, что текст хранится в поле 'text' или 'content'
                text = metadata.get('text') or metadata.get('content') or f"Документ ID: {match.id}"
                contexts.append(text)
                source = metadata.get('source', 'Неизвестный источник')
                sources.append(source)
        else:
            logger.info("   Ничего не найдено в базе знаний.")
            contexts.append("Извините, в базе знаний не найдено информации по вашему вопросу.")
            sources.append("База знаний")

        # 6. Генерация ответа с помощью модели Gemini
        logger.debug("💬 Генерация ответа с помощью Gemini...")
        context_text = "\n\n".join(contexts)

        # Убедитесь, что промпт соответствует ожиданиям модели
        prompt = f"""
        Ты — медицинский ассистент. Отвечай на вопрос, опираясь ТОЛЬКО на предоставленный контекст.
        Отвечай ясно, точно и по существу.
        Если ответа нет в контексте, скажи: "Я не могу дать медицинскую консультацию на основе предоставленных данных. Обратитесь к врачу."

        Контекст:
        {context_text}

        Вопрос: {question}
        Ответ:
        """.strip()

        gemini_response = gemini_model.generate_content(prompt)
        answer = gemini_response.text.strip()
        logger.debug("   Ответ сгенерирован.")

        return AnswerResponse(
            question=question,
            answer=answer,
            sources=sources
        )

    except HTTPException:
        # Перебрасываем HTTPException как есть
        raise
    except Exception as e:
        # Логируем полную трассировку стека для диагностики
        logger.error(f"💥 Неожиданная ошибка в /ask: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        # Возвращаем 500 с деталями (в production лучше скрыть детали)
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

# --- Запуск сервера ---
if __name__ == "__main__":
    import uvicorn
    # Получаем порт из переменной окружения, установленной Cloud Run
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"🚀 Запуск сервера Uvicorn на порту {port}...")
    # ВАЖНО: host должен быть "0.0.0.0" для Cloud Run
    uvicorn.run(app, host="0.0.0.0", port=port)
