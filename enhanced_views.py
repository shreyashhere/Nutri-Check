from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from .models import IngredientAnalysis
from PIL import Image
import numpy as np
import os
# Import with error handling for compatibility
try:
    from langchain_groq import ChatGroq
    ChatGroq = None  # Disable for now due to Pydantic v2 compatibility issues
except ImportError:
    ChatGroq = None

# Alternative approach using groq directly
try:
    import groq
    from langchain.llms.base import LLM
    from typing import Any, List, Optional
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    
    class GroqLLM(LLM):
        """Custom Groq LLM wrapper to avoid Pydantic v2 compatibility issues"""
        
        client: Any = None
        model_name: str = "llama-3.1-8b-instant"
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        @property
        def _llm_type(self) -> str:
            return "groq"
        
        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=4000,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error calling Groq API: {str(e)}")
                return f"Error: Unable to process request. {str(e)}"
    
    GroqAvailable = True
except ImportError:
    GroqAvailable = False
    GroqLLM = None

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from paddleocr import PaddleOCR
from dotenv import load_dotenv
from .rag_service import RAGService
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG service
rag_service = None

def get_rag_service():
    global rag_service
    if rag_service is None:
        try:
            rag_service = RAGService()
            logger.info("RAG service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG service: {str(e)}")
            rag_service = None
    return rag_service

# Lazy initialization of API keys
def get_api_keys():
    langchain_key = os.getenv("LANGCHAIN_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not groq_key:
        raise ValueError("GROQ_API_KEY environment variable is required")
    
    if not langchain_key:
        raise ValueError("LANGCHAIN_API_KEY environment variable is required")
    
    os.environ["LANGCHAIN_API_KEY"] = langchain_key
    os.environ["GROQ_API_KEY"] = groq_key
    
    return groq_key, langchain_key

# Initialize model only when needed
model = None
parser = StrOutputParser()

def get_model():
    global model
    if model is None:
        get_api_keys()  # This will validate keys
        if not GroqAvailable or GroqLLM is None:
            raise ImportError("Groq is not available. Please check your installation.")
        model = GroqLLM()
    return model

# Enhanced system template with RAG integration
enhanced_system_template = '''As a health analysis expert with access to medical research, analyze {category} ingredients from this list: {list_of_ingredients} while considering with STRICT adherence to:
- User allergies: {allergies}
- User medical history: {diseases}

**IMPORTANT: Only proceed with analysis if valid ingredients are detected and category is appropriate. If no valid ingredients are found or category is incorrect, respond with: "Since no valid ingredients are detected for this category, there are no risks specific to the user's profile."**

**Research Context:**
{rag_context}

**Structured Analysis Framework:**

1. **Key Ingredient Analysis** (Focus on 4-5 most significant):
    For each impactful ingredient:
    - Primary use in {category}
    - Benefits (if any) - cite research when available
    - Risks (prioritize allergy/condition conflicts) - cite research when available
    - Safety status vs daily limits - reference medical literature

2. **Personalized Health Impact** ⚠️:
    - Top 3 risks specific to user's profile based on research:
      - Frequency of use
      - Quantity in product
      - Medical history interactions
      - Cite relevant studies when available
      
3. **Should Take or Not 🔍:
    - Ingredients list which are dangerous for user's allergies and conditions:
    - Final recommendation with research backing: Should user take this product or not
    
4. **Smart Alternatives** 💡:
    - 2-3 safer options avoiding flagged ingredients
    - Benefits for user's specific needs
    - Category-appropriate substitutions
    - Reference research for alternatives

5. **Research-Based Insights** 📚:
    - Recent studies or findings related to these ingredients
    - Clinical trial results if available
    - Expert recommendations from medical literature

Format concisely using bullet points, warning symbols(❗), and prioritize medical-critical information. When citing research, mention the source. Ignore unrecognized/unimportant ingredients.'''

enhanced_prompt_template = ChatPromptTemplate.from_messages([("system", enhanced_system_template)])

class OCRReader:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(OCRReader, cls).__new__(cls, *args, **kwargs)
            cls._instance.reader = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
        return cls._instance

    def read_text(self, img):
        results = self.reader.ocr(img, cls=True)
        text_list = []
        if results and results[0]:
            for line in results[0]:
                if len(line) > 1:
                    text_list.append(line[1][0])
        return text_list

ocr_reader = OCRReader()

@csrf_exempt
def analyze_ingredients_with_rag(request):
    """Enhanced ingredient analysis with RAG integration"""
    if request.method == "POST":
        image = request.FILES.get("image")
        category = request.POST.get("category")

        if image and category:
            # Save the uploaded image
            analysis = IngredientAnalysis.objects.create(
                user=request.user, 
                category=category, 
                image=image,
                result=""  # Will be updated after processing
            )

            try:
                img = Image.open(image)
                img = np.array(img)

                # Use the OCRReader class
                results = ocr_reader.read_text(img)
                text_only = [item for item in results if isinstance(item, str)]
                
                # Add debugging
                logger.info(f"OCR Results: {text_only}")
                
                try:
                    allergies = request.user.medicalhistory.allergies.split(',') if hasattr(request.user, 'medicalhistory') and request.user.medicalhistory.allergies else ["No allergy"]
                    diseases = request.user.medicalhistory.diseases.split(',') if hasattr(request.user, 'medicalhistory') and request.user.medicalhistory.diseases else ["No disease"]
                except Exception as e:
                    logger.error(f"Medical history error: {e}")
                    allergies = ["No allergy"]
                    diseases = ["No disease"]
                
                # Join the text list into a string for better processing
                ingredients_text = ", ".join(text_only) if text_only else "No text detected"
                
                # Get RAG service and retrieve relevant information
                rag_service_instance = get_rag_service()
                rag_context = "No research context available."
                
                if rag_service_instance:
                    try:
                        user_profile = {
                            'allergies': ", ".join(allergies),
                            'diseases': ", ".join(diseases)
                        }
                        
                        # Retrieve relevant research information
                        relevant_docs = rag_service_instance.retrieve_relevant_info(
                            ingredients_text, user_profile, k=5
                        )
                        
                        # Format RAG context
                        rag_context = rag_service_instance.format_rag_context(relevant_docs)
                        logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
                        
                    except Exception as e:
                        logger.error(f"RAG retrieval error: {str(e)}")
                        rag_context = "Research context unavailable due to technical issues."
                else:
                    logger.warning("RAG service not available, proceeding without research context")
                
                # Get model with API key validation
                model_instance = get_model()
                
                # Format the prompt manually for our custom LLM
                formatted_prompt = enhanced_system_template.format(
                    list_of_ingredients=ingredients_text,
                    category=category,
                    allergies=", ".join(allergies),
                    diseases=", ".join(diseases),
                    rag_context=rag_context
                )
                
                # Generate enhanced response with RAG context
                llm_response = model_instance._call(formatted_prompt)
                
                # Update the analysis with the enhanced result
                analysis.result = llm_response
                analysis.save()

                # Return the result as JSON
                return JsonResponse({
                    "result": llm_response, 
                    "analysis_id": analysis.id,
                    "rag_used": rag_service_instance is not None,
                    "documents_retrieved": len(relevant_docs) if rag_service_instance else 0
                })

            except Exception as e:
                logger.error(f"Processing error: {str(e)}")
                analysis.delete()  # Clean up if processing fails
                return JsonResponse({"error": f"Processing failed: {str(e)}"}, status=500)

        return JsonResponse({"error": "Invalid input"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)

@login_required
def rag_status(request):
    """Check RAG service status"""
    rag_service_instance = get_rag_service()
    
    status_info = {
        "rag_available": rag_service_instance is not None,
        "vectorstore_loaded": rag_service_instance.vectorstore is not None if rag_service_instance else False,
        "docs_path": rag_service_instance.docs_path if rag_service_instance else None,
        "persist_directory": rag_service_instance.persist_directory if rag_service_instance else None
    }
    
    return JsonResponse(status_info)

@login_required
def refresh_rag_knowledge_base(request):
    """Refresh the RAG knowledge base"""
    try:
        rag_service_instance = get_rag_service()
        if rag_service_instance:
            rag_service_instance.refresh_knowledge_base()
            return JsonResponse({"status": "success", "message": "Knowledge base refreshed successfully"})
        else:
            return JsonResponse({"status": "error", "message": "RAG service not available"}, status=500)
    except Exception as e:
        logger.error(f"Error refreshing knowledge base: {str(e)}")
        return JsonResponse({"status": "error", "message": f"Error: {str(e)}"}, status=500)

@login_required
def get_ingredient_research(request, ingredient_name):
    """Get specific research information about an ingredient"""
    try:
        rag_service_instance = get_rag_service()
        if rag_service_instance:
            docs = rag_service_instance.get_ingredient_specific_info(ingredient_name)
            research_info = rag_service_instance.format_rag_context(docs)
            return JsonResponse({
                "ingredient": ingredient_name,
                "research_info": research_info,
                "sources_count": len(docs)
            })
        else:
            return JsonResponse({"error": "RAG service not available"}, status=500)
    except Exception as e:
        logger.error(f"Error getting ingredient research: {str(e)}")
        return JsonResponse({"error": f"Error: {str(e)}"}, status=500)

@login_required
def rag_dashboard(request):
    """RAG system dashboard view"""
    return render(request, 'rag_dashboard.html')
