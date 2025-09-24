import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# LangChain and Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

# --- 1. INITIALIZE FLASK APP ---
app = Flask(__name__)
CORS(app)

# Global variable to hold the loaded agent
agent_executor = None

# --- 2. CROP PARAMETERS DATABASE ---
CROP_PARAMETERS = {
    'rice': {'water_per_hectare_liters': 12_000_000, 'nitrogen_per_tonne': 20.0},
    'maize': {'water_per_hectare_liters': 6_000_000, 'nitrogen_per_tonne': 22.0},
    'cotton': {'water_per_hectare_liters': 8_480_000, 'nitrogen_per_tonne': 25.0},
    'pearl_millet': {'water_per_hectare_liters': 3_230_000, 'nitrogen_per_tonne': 15.0},
    'finger_millet': {'water_per_hectare_liters': 3_230_000, 'nitrogen_per_tonne': 15.0},
    'green_gram': {'water_per_hectare_liters': 3_246_000, 'nitrogen_per_tonne': 20.0},
    'black_gram': {'water_per_hectare_liters': 2_700_000, 'nitrogen_per_tonne': 18.0},
    'groundnut': {'water_per_hectare_liters': 2_700_000, 'nitrogen_per_tonne': 18.0},
    'sesame': {'water_per_hectare_liters': 4_900_000, 'nitrogen_per_tonne': 20.0},
    'mustard': {'water_per_hectare_liters': 4_000_000, 'nitrogen_per_tonne': 22.0},
    'lentil': {'water_per_hectare_liters': 3_500_000, 'nitrogen_per_tonne': 20.0},
    'default': {'water_per_hectare_liters': 5_500_000, 'nitrogen_per_tonne': 15.0}
}

# --- 3. RECOMMENDATION ENGINE ---
def generate_recommendations(crop, area, target_yield):
    params = CROP_PARAMETERS.get(crop.lower(), CROP_PARAMETERS['default'])
    total_water = params['water_per_hectare_liters'] * area
    total_nitrogen = target_yield * area * params['nitrogen_per_tonne']
    pesticide_recommendation = "Follow Integrated Pest Management (IPM) practices. Apply as needed."

    report = f"""=== Crop Optimization Report for {crop.title()} ===

- Use {total_water:,.0f} liters of water
- Apply {total_nitrogen:.2f} kg of nitrogen fertilizer
- {pesticide_recommendation}

🌱 Optimized for estimated yield of {target_yield:.2f} tonnes per hectare."""

    if total_water > 10_000_000:
        report += "\n💧 Note: Water usage is high. Consider irrigating early morning or evening."
    return report

# --- 4. INITIALIZE AI AGENT ---
def initialize_agent():
    global agent_executor
    google_key = os.getenv("GOOGLE_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not google_key:
        print("❌ ERROR: GOOGLE_API_KEY not found in environment variables.")
        return

    try:
        print("Initializing LLM with API key...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            api_key=google_key  # Pass API key explicitly
        )

        tools = []
        if tavily_key:
            tools.append(TavilySearchResults(max_results=3))

        print("Pulling prompt from LangChain Hub...")
        prompt = hub.pull("hwchase17/react")

        print("Creating agent...")
        agent = create_react_agent(llm, tools, prompt)

        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        print("✅ AI Agent initialized successfully.")
    except Exception as e:
        print(f"❌ ERROR initializing agent: {e}")

# --- 5. API ENDPOINTS ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "✅ AgriGem API is running!",
        "endpoints": {
            "ask": "/ask (POST with {question})",
            "recommend": "/recommend (POST with {crop, area, target_yield})"
        }
    })

@app.route('/ask', methods=['POST'])
def ask_gem():
    if not agent_executor:
        return jsonify({"error": "AI Agent not available. Please check GOOGLE_API_KEY and redeploy."}), 503
    data = request.get_json(silent=True)
    question = data.get('question') if data else None
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        response = agent_executor.invoke({"input": question})
        return jsonify({"answer": response.get('output', 'No response generated')})
    except Exception as e:
        return jsonify({'error': f'Agent execution failed: {str(e)}'}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON input"}), 400
    crop = data.get('crop')
    area = data.get('area')
    target_yield = data.get('target_yield')
    if not crop or area is None or target_yield is None:
        return jsonify({"error": "Missing required fields (crop, area, target_yield)"}), 400
    try:
        area = float(area)
        target_yield = float(target_yield)
        report_text = generate_recommendations(crop, area, target_yield)
        return jsonify({"report": report_text})
    except ValueError:
        return jsonify({"error": "Invalid numeric values for area or target_yield"}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 6. RUN SERVER ---
if __name__ == '__main__':
    initialize_agent()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
