CHROMA_PATH = "chroma"
DATA_PATH = "data"


text_prompt = """Based on the following context, generate 50 precise rules for growth-maintenance-attitude laws for '{query}' strictly in the format 'if <condition> then <do>'.

The context contains information about '{query}':
{context}

IMPORTANT: 
- Focus on MEASURABLE conditions and QUANTIFIABLE actions wherever possible
- Your rules MUST reference information from at least 10 different sources in the context
- If fewer than 10 sources are available, extract multiple distinct pieces of information from each source

Example format:
if the humidity falls below 40% then spray the leaves with 100ml water daily
if the soil pH rises above 7.5 then add 5g of sulfur per 1m² of soil
if the temperature drops below 15°C then reduce watering to 50ml per week
if leaf yellowing affects more than 20% of the plant then reduce nitrogen fertilizer by 50%

Your task:
1. FIRST identify at least 10 distinct sources or information clusters within the context
2. Extract SPECIFIC MEASUREMENTS from each source (temperatures, humidity levels, pH values, nutrient concentrations, time periods, etc.)
3. Include PRECISE NUMBERS in both conditions and actions (exact temperatures, volumes, weights, percentages, frequencies, durations)
4. Use CONSISTENT UNITS throughout all rules (choose metric units: ml, g, cm, m², °C, etc.)
5. For each threshold condition, specify a clear numerical value
6. For each recommended action, include specific quantities or percentages
7. Generate all 50 rules using the same units and measurement systems
8. Ensure the rules collectively cover information from at least 10 different sources or information segments
9. Prioritize measurable conditions over subjective observations

List ONLY the rules, without introductions or explanations."""