CHROMA_PATH = "chroma"
DATA_PATH = "data"


text_prompt = """Based on the following context, generate 10 precise rules for growth-maintenance-attitude laws for '{query}' strictly in the format.

The context contains information about '{query}':
{context}

IMPORTANT: 
- Focus on MEASURABLE conditions and QUANTIFIABLE actions wherever possible
- Your rules MUST reference information from at least 5 different sources in the context
- Input: sensors: humidity, temperature
- Output: controls: ventilator speed

format:
{{
  "rules": [
    {{
      "id": "rule1",
      "description": "",
      "conditions": [
        {{
          "parameter": "",
          "operator": "",
          "value": ,
          "unit": ""
        }}
      ],
      "actions": [
        {{
          "parameter": "",
          "value": ,
          "unit": ""
        }}
      ]
    }},
    {{
      "id": "rule2",
      "description": "",
      "conditions": [
        {{
          "parameter": "",
          "operator": "",
          "value": ,
          "unit": ""
        }}
      ],
      "actions": [
        {{
          "parameter": "",
          "value": ""
        }}
      ]
    }},
    {{
      "id": "rule3",
      "description": "",
      "conditions": [
        {{
          "parameter": "",
          "operator": "",
          "value": ,
          "unit": ""
        }}
      ],
      "actions": [
        {{
          "parameter": "",
          "value": ""
        }}
      ]
    }}
    // Additional rules can be added here
  ]
}}

Your task:
1. FIRST identify at least 10 distinct sources or information clusters within the context
2. Extract SPECIFIC MEASUREMENTS from each source (temperatures, humidity levels)
3. Include PRECISE NUMBERS in both conditions and actions (exact temperatures)
4. Use CONSISTENT UNITS throughout all rules (choose metric units:°C)
5. For each threshold condition, specify a clear numerical value
6. For each recommended action, include specific quantities or percentages
7. Generate all 10 rules using the same units and measurement systems
8. Ensure the rules collectively cover information from at least 5 different sources or information segments
9. Prioritize measurable conditions over subjective observations


List ONLY the rules, without introductions or explanations."""