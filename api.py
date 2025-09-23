import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64

# LangChain and Gemini Imports
# ... (all your existing imports are the same)
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage


# --- 1. INITIALIZE FLASK APP AND GLOBAL VARIABLES ---
app = Flask(__name__)
CORS(app)

rag_chain = None
llm = None

# --- 2. EXPANDED CROP PARAMETERS DATABASE ---
# This dictionary holds the specific rules for each major crop in Odisha.
# These values are based on general agricultural science and can be further refined.
CROP_PARAMETERS = {
    # Kharif Cereals
    'rice': { 'water_per_hectare_liters': 12_000_000, 'nitrogen_per_tonne': 20.0 },
    'maize': { 'water_per_hectare_liters': 6_000_000, 'nitrogen_per_tonne': 22.0 },
    'ragi': { 'water_per_hectare_liters': 4_500_000, 'nitrogen_per_tonne': 20.0 },
    'jowar': { 'water_per_hectare_liters': 5_000_000, 'nitrogen_per_tonne': 18.0 },
    'bajra': { 'water_per_hectare_liters': 4_000_000, 'nitrogen_per_tonne': 15.0 },
    
    # Kharif Pulses (Legumes often have lower external N needs but are protein-rich)
    'arhar/tur': { 'water_per_hectare_liters': 4_000_000, 'nitrogen_per_tonne': 40.0 },
    'moong': { 'water_per_hectare_liters': 3_500_000, 'nitrogen_per_tonne': 35.0 },
    'urad': { 'water_per_hectare_liters': 3_500_000, 'nitrogen_per_tonne': 35.0 },

    # Kharif Oilseeds
    'groundnut': { 'water_per_hectare_liters': 5_000_000, 'nitrogen_per_tonne': 40.0 },
    'sesamum': { 'water_per_hectare_liters': 4_000_000, 'nitrogen_per_tonne': 30.0 },
    
    # Kharif Fibres & Others
    'cotton': { 'water_per_hectare_liters': 8_000_000, 'nitrogen_per_tonne': 40.0 },
    'jute': { 'water_per_hectare_liters': 10_000_000, 'nitrogen_per_tonne': 10.0 },
    'sugarcane': { 'water_per_hectare_liters': 20_000_000, 'nitrogen_per_tonne': 3.0 },

    # Rabi Crops
    'wheat': { 'water_per_hectare_liters': 4_000_000, 'nitrogen_per_tonne': 25.0 },
    'mustard': { 'water_per_hectare_liters': 3_500_000, 'nitrogen_per_tonne': 40.0 },
    'gram': { 'water_per_hectare_liters': 3_000_000, 'nitrogen_per_tonne': 45.0 }, # Chickpea
    
    # Default values for any other crop not listed
    'default': {
        'water_per_hectare_liters': 5_000_000,
        'nitrogen_per_tonne': 15.0
    }
}

# --- 3. RECOMMENDATION ENGINE LOGIC ---
# This function is now much more powerful and requires no changes.
def generate_recommendations(crop, area, target_yield):
    crop_key = crop.lower().replace(" ", "") # Standardize the crop name
    params = CROP_PARAMETERS.get(crop_key, CROP_PARAMETERS['default'])
    
    total_water = params['water_per_hectare_liters'] * area
    total_nitrogen = target_yield * area * params['nitrogen_per_tonne']
    pesticide_recommendation = "Follow Integrated Pest Management (IPM) practices. Apply as needed based on scouting."

    report = f"""=== Crop Optimization Report for {crop.title()} ===

- Use {total_water:,.0f} liters of water
- Apply {total_nitrogen:.2f} kg of nitrogen fertilizer
- {pesticide_recommendation}

🌱 This plan is optimized for an estimated yield of {target_yield:.2f} tonnes per hectare.
"""
    if total_water > 10_000_000:
        report += "\n💧 Note: Water usage is high. Consider irrigating during the early morning or evening to reduce evaporation."
    return report

# --- 4. SETUP FUNCTION TO LOAD KNOWLEDGE BASE ---
# (This function remains the same as before)
def initialize_chatbot():
    global rag_chain, llm
    # ... (rest of the initialization code is unchanged)
    print("--- Initializing Gemini Gem Chatbot ---")
    load_dotenv()
    try:
        print("\nLoading knowledge base...")
        excel_loader = UnstructuredExcelLoader("knowledge_base/odisha_stats.xlsx", mode="single")
        documents = excel_loader.load()
        print(f"Loaded {len(documents)} documents into the knowledge base.")
        print("Creating vector embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(documents, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        template = "You are a helpful assistant for farmers in Odisha. Answer based ONLY on the provided context.\nCONTEXT: {context}\nQUESTION: {question}\nANSWER:"
        prompt = PromptTemplate.from_template(template)
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("✅ Gemini Gem (RAG Chain) initialized successfully.")
    except Exception as e:
        print(f"❌ ERROR initializing Gem: {e}")

# --- 5. CREATE THE API ENDPOINTS ---
# (The API endpoints remain the same as before)
@app.route('/ask', methods=['POST'])
def ask_gem():
    # ... (code is unchanged)
    if not llm:
        return jsonify({"error": "Chatbot is not initialized. Check server logs."}), 503
    question = request.form.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        image_file = request.files.get('image')
        if image_file:
            print("Processing a request with an image.")
            image_bytes = image_file.read()
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            image_part = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            text_part = {"type": "text", "text": f"You are an expert agricultural assistant. Analyze the attached image of a crop and answer the user's question: '{question}'. Identify potential diseases, pests, or nutrient deficiencies visible in the image and suggest solutions."}
            message = HumanMessage(content=[text_part, image_part])
            response = llm.invoke([message])
            answer = response.content
        else:
            print("Processing a text-only request.")
            answer = rag_chain.invoke(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    # ... (code is unchanged)
    try:
        data = request.get_json()
        crop = data.get('crop')
        area = float(data.get('area'))
        target_yield = float(data.get('target_yield'))
        if not all([crop, area, target_yield]):
            return jsonify({"error": "Missing required fields: crop, area, target_yield"}), 400
        report_text = generate_recommendations(crop, area, target_yield)
        return jsonify({"report": report_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 6. RUN THE SERVER ---
if __name__ == '__main__':
    initialize_chatbot()
    app.run(port=5001, debug=True)