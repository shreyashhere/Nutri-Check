# ingredient_analysis_app/views.py
# import pytesseract
# import cv2
# import requests
# import re

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
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# import easyocr
from paddleocr import PaddleOCR
from dotenv import load_dotenv
load_dotenv()

# from django.utils.html import format_html
# Specify the full path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Lazy initialization of API keys - only validate when actually needed
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
        model = ChatGroq(model="llama3-8b-8192")
    return model

# Create the prompt template
system_template = '''As a health analysis expert, analyze {category} ingredients from this list: {list_of_ingredients} while considering with STRICT adherence to:
- User allergies: {allergies}
- User medical history: {diseases}

**IMPORTANT: Only proceed with analysis if valid ingredients are detected and category is appropriate. If no valid ingredients are found or category is incorrect, respond with: "Since no valid ingredients are detected for this category, there are no risks specific to the user's profile."**

**Structured Analysis Framework:**

1. **Key Ingredient Analysis** (Focus on 4-5 most significant):
    For each impactful ingredient:
    - Primary use in {category}
    - Benefits (if any)
    - Risks (prioritize allergy/condition conflicts)
    - Safety status vs daily limits

2. **Personalized Health Impact** ⚠️:
    - Top 3 risks specific to user's profile :
      - Frequency of use
      - Quantity in product
      - Medical history interactions
      
3. **Should Take or Not 🔍:
    - Ingredients list which are dangerous for user's allergies and conditions :
    - Final recommendation, Should user take this product or not: 
    
4. **Smart Alternatives** 💡:
    - 2-3 safer options avoiding flagged ingredients
    - Benefits for user's specific needs
    - Category-appropriate substitutions

Format concisely using bullet points, warning symbols(❗), and prioritize medical-critical information. Ignore unrecognized/unimportant ingredients.'''

prompt_template = ChatPromptTemplate.from_messages([("system", system_template)])
# ans = "**Ingredient Analysis:**\n\n1. **WATER**: Common use: hydration. Positive effects: essential for human body, helps regulate body temperature, and transports nutrients. Negative effects: excessive consumption can lead to water intoxication.\n2. **CARBONATED BEVERAGET**: Common use: carbonation in beverages. Positive effects: adds fizz, helps with digestion. Negative effects: can lead to bloating, gas, and stomach discomfort.\n3. **CARBONATED**: Common use: same as above. No specific health effects.\n4. **ACIDITY REGULATOR**: Common use: adjusts pH levels in foods and beverages. Positive effects: prevents spoilage, preserves food. Negative effects: can disrupt gut health, lead to digestive issues.\n5. **TANS PERMITTED NATURAL**: Common use: natural coloring agent. Positive effects: no specific health effects. Negative effects: can cause allergic reactions.\n6. **FLAVOURS**: Common use: adds taste and aroma to products. Positive effects: enhances flavor. Negative effects: can be overwhelming, disrupt digestive system.\n7. **CONTAINS NO**: Common use: labeling. No specific health effects.\n8. **CAFFEINE**: Common use: stimulant in beverages. Positive effects: improves alertness, boosts energy. Negative effects: can lead to addiction, insomnia, and increased heart rate.\n9. **PROTEIN: 0 g**: Common use: labeling. No specific health effects.\n10. **SUGAR**: Common use: sweetener. Positive effects: provides energy. Negative effects: contributes to obesity, diabetes, and tooth decay.\n\n**Health Impact Prediction:**\n\nThe cumulative health impact of daily usage of Coke is likely to be negative, considering the high sugar content, caffeine, and acidity regulator. Consuming large amounts of sugar and caffeine can lead to increased risk of chronic diseases, such as diabetes, heart disease, and certain types of cancer.\n\n**Healthier Alternatives:**\n\n1. **Infused water**: Instead of Coke, try infused water with natural flavors and no added sugars.\n2. **Herbal teas**: Herbal teas, such as peppermint or chamomile, can provide a caffeine-free alternative to Coke.\n3. **Low-calorie beverages**: Opt for low-calorie, sugar-free beverages with natural sweeteners like stevia or erythritol.\n\n**Summary:**\n\nCoke is not suitable for anyone, especially those with sugar or caffeine sensitivities. It's essential to consider individual health needs and preferences when choosing a beverage. If you're looking for a healthier alternative, consider infused water, herbal teas, or low-calorie beverages.\n\n**Product Name:** Coca-Cola (Coke)"



# Ensure the user is logged in
def home(request):
    return render(request, "home.html")


@login_required
def upload(request):
    return render(request, "ingredient_analysis_app/upload.html")


class OCRReader:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(OCRReader, cls).__new__(cls, *args, **kwargs)
            cls._instance.reader =  PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
        return cls._instance

    def read_text(self, img):
        # PaddleOCR returns different format: [[[bbox], (text, confidence)]]
        results = self.reader.ocr(img, cls=True)
        # Extract only text from PaddleOCR results
        text_list = []
        if results and results[0]:
            for line in results[0]:
                if len(line) > 1:
                    text_list.append(line[1][0])  # line[1][0] contains the text
        return text_list

ocr_reader = OCRReader()

@csrf_exempt
def analyze_ingredients(request):
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
                print(f"OCR Results: {text_only}")
                
                try:
                    allergies = request.user.medicalhistory.allergies.split(',') if hasattr(request.user, 'medicalhistory') and request.user.medicalhistory.allergies else ["No allergy"]
                    diseases = request.user.medicalhistory.diseases.split(',') if hasattr(request.user, 'medicalhistory') and request.user.medicalhistory.diseases else ["No disease"]
                except Exception as e:
                    print(f"Medical history error: {e}")
                    allergies = ["No allergy"]
                    diseases = ["No disease"]
                
                # Join the text list into a string for better processing
                ingredients_text = ", ".join(text_only) if text_only else "No text detected"
                
                # Get model with API key validation
                model_instance = get_model()
                chain = prompt_template | model_instance | parser
                llm_response = chain.invoke(
                    {
                        "list_of_ingredients": ingredients_text, 
                        "category": category, 
                        "allergies": ", ".join(allergies), 
                        "diseases": ", ".join(diseases)
                    }
                )
                
                # Update the analysis with the result
                analysis.result = llm_response
                analysis.save()

                # Return the result as JSON
                return JsonResponse({"result": llm_response, "analysis_id": analysis.id})

            except Exception as e:
                print(f"Processing error: {str(e)}")  # Add debugging
                analysis.delete()  # Clean up if processing fails
                return JsonResponse({"error": f"Processing failed: {str(e)}"}, status=500)

        return JsonResponse({"error": "Invalid input"}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)


def history(request):
    # Retrieve all analyses related to the logged-in user
    user_analyses = IngredientAnalysis.objects.filter(user=request.user).order_by(
        "-timestamp"
    )
    return render(request, "history.html", {"user_analyses": user_analyses})


def analysis_detail(request, analysis_id):
    analysis = IngredientAnalysis.objects.get(id=analysis_id, user=request.user)
    return render(request, "analysis_detail.html", {"analysis": analysis})


def register(request):
    if request.method == "POST":
        username = request.POST["username"]
        email = request.POST["email"]
        password = request.POST["password"]
        repeat_password = request.POST["repeatPassword"]

        if password != repeat_password:
            return render(
                request, "register.html", {"error_message": "Passwords don't match"}
            )

        if User.objects.filter(username=username).exists():
            return render(
                request, "register.html", {"error_message": "Username already taken"}
            )

        user = User.objects.create_user(
            username=username, email=email, password=password
        )
        user.save()

        login(request, user)
        return redirect("check_medical")
        # messages.success(request, 'Account created successfully! Please log in.')
        # return redirect('login')

    return render(request, "register.html")


def user_login(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("check_medical")
        else:
            return render(
                request, "login.html", {"error_message": "Invalid credentials"}
            )

    return render(request, "login.html")


def user_logout(request):
    logout(request)
    return redirect("home")



# Convert the image to grayscale and preprocess it for OCR
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# sharpened = cv2.filter2D(blurred, -1, kernel)
# binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# # Use Tesseract to extract text from the image
# custom_config = r'--oem 3 --psm 6'
# extracted_text = pytesseract.image_to_string(binary, config=custom_config)

'''
    url = "https://api.ocr.space/parse/image"
    payloads = [
        {
            "apikey": "K85352834688957",  
            "language": "eng",
        },
        {
            "apikey": "K86276895488957", 
            "language": "eng",
        },
    ]
    payload = payloads[0]
    files = {"file": image}
    response = requests.post(url, data=payload, files=files)
    results = response.json()

    if results["IsErroredOnProcessing"]:
        return JsonResponse(
            {
                "error": "Image size exceeds the maximum permissible file size limit of 1024 KB"
            },
            status=400,
        )
    elif (
        results
        == "For this API KEY only 300 concurrent connections at the same time allowed. Contact support if you need more."
    ):
        payloads = payloads[1]
        response = requests.post(url, data=payload, files=files)
        results = response.json()
        
        if results["IsErroredOnProcessing"]:
            return JsonResponse(
                {
                    "error": "Image size exceeds the maximum permissible file size limit of 1024 KB"
                },
                status=400,
            )
        elif results == "For this API KEY only 300 concurrent connections at the same time allowed. Contact support if you need more.":
            return JsonResponse(
                {"error": "Image parsing is down... Try again after some time"}, status=400
            )

    # Generate LLM analysis
'''