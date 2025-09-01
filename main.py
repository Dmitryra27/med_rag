# main.py для Cloud Run (с Pinecone и Vertex AI)
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import pinecone  # Импортируем Pinecone
from fastapi import Response
from google.cloud import storage
from google.oauth2 import service_account
import uvicorn
from datetime import datetime
from google.auth import default
from google.cloud import aiplatform
from fastapi.middleware.cors import CORSMiddleware
# Добавьте в начало файла, если еще не импортировано
import logging

# --- Глобальные переменные для хранения инициализированных клиентов ---
embedding_model = None
gemini_model = None
pinecone_index = None

# --- Конфигурация ---
PROJECT_ID = os.environ.get("PROJECT_ID", "ai-project-26082025")
REGION = os.environ.get("REGION", "us-central1")
# --- Конфигурация Pinecone ---
# Убедитесь, что эти переменные установлены в Cloud Run
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
#print("🚀  API KEY...", PINECONE_API_KEY)
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "med-index")  # Используем имя из примера Pinecone


def setup_authentication():
    """Настройка аутентификации для локальной отладки"""
    try:
        # Попробуем получить учетные данные по умолчанию
        credentials, project = default()
        print(f"✅ Аутентификация успешна. Проект: {project}")

        # Установим переменные окружения если они не установлены
        if not os.environ.get('GOOGLE_CLOUD_PROJECT'):
            os.environ['GOOGLE_CLOUD_PROJECT'] = project or 'ai-project-26082025'

        if not os.environ.get('GOOGLE_CLOUD_REGION'):
            os.environ['GOOGLE_CLOUD_REGION'] = 'us-central1'

        return True
    except Exception as e:
        print(f"❌ Ошибка аутентификации: {e}")
        print("Попробуйте выполнить: gcloud auth application-default login")
        return False




# --- Lifespan handler для FastAPI ---
# Это гарантирует, что инициализация произойдет при старте приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Инициализация при запуске ---
    print("🚀 Запуск приложения Medical RAG API...")

    # 1. Инициализация Vertex AI (обязательно до импорта моделей!)
    # Убедитесь, что PROJECT_ID и REGION установлены правильно и сервисный аккаунт имеет доступ
    try:
        credentials = service_account.Credentials.from_service_account_file("ai-project-26082025-6a65bc63db88.json")
        storage_client = storage.Client(credentials=credentials)

        vertexai.init(project=PROJECT_ID, location=REGION)

        print("✅ Vertex AI инициализирована.")
    except Exception as e:
        print(f"❌ Ошибка инициализации Vertex AI: {e}")
        # В production лучше бросить исключение, чтобы сервис не стартовал
        # raise
        # Но для демонстрации продолжим
        pass

    # 2. Инициализация моделей Vertex AI
    global embedding_model, gemini_model
    try:
        # Убедитесь, что модель существует и доступна в вашем регионе
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        # Используем более стабильную и быструю модель для генерации
        gemini_model = GenerativeModel("gemini-2.5-pro")
        print("✅ Модели Vertex AI загружены.")
    except Exception as e:
        print(f"❌ Ошибка загрузки моделей Vertex AI: {e}")
        # Критично, без моделей сервис не работает
        raise

    # 3. Инициализация Pinecone
    global pinecone_index
    try:
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY environment variable is not set.")
        if not PINECONE_INDEX_NAME:
            raise ValueError("PINECONE_INDEX_NAME environment variable is not set.")

        # Инициализация Pinecone (новый SDK v5+)
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        # Подключаемся к индексу
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        # Проверка: получим статистику индекса
        index_stats = pinecone_index.describe_index_stats()
        print(
            f"✅ Pinecone инициализирован. Индекс '{PINECONE_INDEX_NAME}' содержит {index_stats.get('total_vector_count', 0)} векторов.")
    except Exception as e:
        print(f"❌ Ошибка инициализации Pinecone: {e}")
        # Критично, без БД сервис не работает
        raise

    print("🟢 Приложение готово к обработке запросов.")
    yield  # Приложение работает
    # --- Очистка при завершении (если нужна) ---
    print("🛑 Завершение работы приложения...")
    # Pinecone SDK не требует явного закрытия


# --- Создание приложения FastAPI с lifespan handler ---
app = FastAPI(lifespan=lifespan, title="Medical RAG API", version="2.0.0-Pinecone")

# Добавьте CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Для разработки разрешаем все origins
    allow_credentials=True,
    allow_methods=["*"],  # Разрешаем все методы (GET, POST, OPTIONS и т.д.)
    allow_headers=["*"],  # Разрешаем все заголовки
    # В production лучше указать конкретные origins:
    # allow_origins=["http://localhost:63342", "https://your-domain.com"],
)

# --- Эндпоинты API ---
@app.get("/")
async def home():
    """Корневой эндпоинт для проверки состояния сервиса."""
    # Получаем статистику Pinecone для здоровья
    try:
        stats = pinecone_index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 'N/A')
    except:
        vector_count = 'Ошибка получения статистики'

    return {
        "status": "ok",
        "message": "Medical RAG API Server on Google Cloud Run (Pinecone + Vertex AI)",
        "version": "2.0.0-Pinecone",
        "pinecone_vectors": vector_count
    }


@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    try:
        # Проверяем Pinecone - получаем статистику индекса
        if pinecone_index is not None:
            stats = pinecone_index.describe_index_stats()
            pinecone_status = "healthy"
            vector_count = stats.get('total_vector_count', 0)
            dimension = stats.get('dimension', 'N/A')
        else:
            pinecone_status = "uninitialized"
            vector_count = 0
            dimension = 'N/A'

    except Exception as e:
        pinecone_status = "unhealthy"
        vector_count = 0
        dimension = 'N/A'
        # Логируем ошибку для отладки
        print(f"Health check - Pinecone error: {e}")

    # Проверяем Vertex AI модели
    embedding_model_status = "healthy" if embedding_model is not None else "uninitialized"
    gemini_model_status = "healthy" if gemini_model is not None else "uninitialized"

    return {
        "status": "healthy" if (
                    pinecone_status == "healthy" and embedding_model_status == "healthy" and gemini_model_status == "healthy") else "degraded",
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


# Добавьте этот код в ваш main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Модель данных для запроса
class QuestionRequest(BaseModel):
    question: str


# Модель данных для ответа
class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest, response: Response):
    """
    Эндпоинт для ответа на медицинские вопросы с использованием RAG
    """
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    try:
        question = request.question.strip()

        # Валидация вопроса
        if not question:
            raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

        if len(question) > 1000:
            raise HTTPException(status_code=400, detail="Вопрос слишком длинный (максимум 1000 символов)")

        logger.info(f"Получен вопрос: {question}")

        # 1. Создание эмбеддинга для вопроса
        if embedding_model is None:
            raise HTTPException(status_code=500, detail="Модель эмбеддингов не инициализирована")

        try:
            embedding_response = embedding_model.get_embeddings([question])
            question_embedding = embedding_response[0].values
            logger.info("Эмбеддинг вопроса создан успешно")
        except Exception as e:
            logger.error(f"Ошибка создания эмбеддинга: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка обработки вопроса: {str(e)}")

        # 2. Поиск похожих документов в Pinecone
        if pinecone_index is None:
            raise HTTPException(status_code=500, detail="Индекс Pinecone не инициализирован")

        try:
            search_results = pinecone_index.query(
                vector=question_embedding,
                top_k=3,  # Количество результатов для контекста
                include_metadata=True
            )
            logger.info(f"Поиск в Pinecone завершен, найдено {len(search_results.matches)} результатов")
        except Exception as e:
            logger.error(f"Ошибка поиска в Pinecone: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка поиска в базе знаний: {str(e)}")

        # 3. Обработка результатов поиска
        contexts = []
        sources = []

        if search_results.matches:
            for match in search_results.matches:
                metadata = match.metadata or {}
                # Извлекаем текст из метаданных (адаптируйте под вашу структуру)
                text = metadata.get('text') or metadata.get('content') or metadata.get(
                    'preview') or f"Документ ID: {match.id}"
                contexts.append(text)

                # Извлекаем источник
                source = metadata.get('source', 'Неизвестный источник')
                sources.append(source)
        else:
            # Если ничего не найдено, используем заглушку
            contexts.append("Извините, но в базе знаний не найдено информации по вашему вопросу.")
            sources.append("База знаний")

        # 4. Генерация ответа с помощью модели Gemini
        if gemini_model is None:
            raise HTTPException(status_code=500, detail="Модель генерации не инициализирована")

        try:
            # Формируем контекст для модели
            context_text = "\n\n".join(contexts)

            # Создаем промпт для модели
            prompt = f"""
            Ты — медицинский ассистент. Отвечай на вопрос, опираясь ТОЛЬКО на предоставленный контекст.
            Отвечай ясно, точно и по существу.
            Если ответа нет в контексте, скажи: "Я не могу дать медицинскую консультацию на основе предоставленных данных. Обратитесь к врачу."

            Контекст:
            {context_text}

            Вопрос: {question}
            Ответ:
            """.strip()

            # Генерируем ответ
            response = gemini_model.generate_content(prompt)
            answer = response.text.strip()

            logger.info("Ответ сгенерирован успешно")

        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            # В случае ошибки генерации возвращаем ответ на основе контекста
            if contexts and contexts[0] != "Извините, но в базе знаний не найдено информации по вашему вопросу.":
                answer = "На основе найденной информации: " + contexts[0][:500] + "..."
            else:
                answer = "Извините, не удалось сгенерировать ответ. Обратитесь к врачу."

        # 5. Возвращаем ответ
        return AnswerResponse(
            question=question,
            answer=answer,
            sources=sources
        )

    except HTTPException:
        # Перебрасываем HTTP исключения как есть
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка в /ask: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


if __name__ == "__main__":
    if not setup_authentication():
        exit(1)
    import uvicorn

    # Получаем порт из переменной окружения, установленной Cloud Run
    port = int(os.environ.get("PORT", 8080))
    print(f"Запуск сервера на порту {port}...")
    # ВАЖНО: host должен быть "0.0.0.0" для Cloud Run
    uvicorn.run(app, host="0.0.0.0", port=port)
