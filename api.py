import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

app = Flask(__name__)
CORS(app)

agent_executor = None

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

def generate_recommendations(crop, area, target_yield):
    params = CROP_PARAMETERS.get(crop.lower(), CROP_PARAMETERS['default'])
    total_water = params['water_per_hectare_liters'] * area
    total_nitrogen = target_yield * area * params['nitrogen_per_tonne']
    report = f"=== Crop Optimization Report for {crop.title()} ===\n\n"
    report += f"- Use {total_water:,.0f} liters of water\n"
    report += f"- Apply {total_nitrogen:.2f} kg of nitrogen fertilizer\n"
    report += "- Follow Integrated Pest Management (IPM) practices. Apply as needed.\n"
    report += f"\n🌱 This plan is optimized for an estimated yield of {target_yield:.2f} tonnes per hectare."
    if total_water > 10_000_000:
        report += "\n💧 Note: Water usage is high. Consider irrigating early morning/evening."
    return report

def initialize_agent():
    global agent_executor
    google_key = os.getenv("GOOGLE_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not google_key:
        print("❌ GOOGLE_API_KEY missing!")
        return

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            api_key=google_key  # pass key explicitly
        )

        tools = []
        if tavily_key:
            tools.append(TavilySearchResults(max_results=3))

        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        print("✅ Web Search Agent initialized successfully.")
    except Exception as e:
        print(f"❌ ERROR initializing agent: {e}")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "✅ AgriGem API is running!",
        "endpoints": {"ask": "/ask", "recommend": "/recommend"}
    })

@app.route('/ask', methods=['POST'])
def ask_gem():
    global agent_executor
    if not agent_executor:
        return jsonify({"error": "AI Agent not available. Please check GOOGLE_API_KEY and redeploy."}), 503
    data = request.get_json(silent=True)
    if not data or not data.get("question"):
        return jsonify({"error": "No question provided"}), 400
    try:
        resp = agent_executor.invoke({"input": data["question"]})
        return jsonify({"answer": resp.get("output", "No response generated")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json(silent=True)
    if not data or not all(k in data for k in ("crop", "area", "target_yield")):
        return jsonify({"error": "Missing required fields"}), 400
    try:
        report = generate_recommendations(data["crop"], float(data["area"]), float(data["target_yield"]))
        return jsonify({"report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    initialize_agent()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
