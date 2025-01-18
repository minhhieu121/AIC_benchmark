import google.generativeai as genai
import json

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

class GeminiReranker:
    def __init__(self):
        self.query = ''
        self.top_k_vid = ''
        self.model_name = ''
        self.api_key = ''
        self.prompt = ''
    
    def __init__(self, query, top_k_vid, model_name, api_key):
        self.query = query
        self.top_k_vid = top_k_vid
        self.model_name = model_name
        self.api_key = api_key
        self.prompt = ''

    def __convert_to_json(self):
        try:
            json_data = json.dumps(self.top_k_vid, indent=2)
            # print(self.top_k_vid)
            return json_data
        except Exception as e:
            print(f"Error converting to JSON: {str(e)}" )
            return None

    def __initialize_reranker(self):
        genai.configure(api_key = self.api_key)
        json_string = self.__convert_to_json()
        self.prompt = f"""
            You are an expert in video content analysis and ranking.

            **MUST** follow the instructions below precisely.

            **User Query:**
            "{self.query}"

            **Video Descriptions:**
            {json_string}

            **Instructions:**
            1. **Analyze** the relevance of each video description to the user query.
            2. Thinking step by step, focus on the state of action, the environment around and specific details
            3. **FORCE** the output to adhere strictly to the following format:
                #video1#{'reason/grading_metrics'}#video2#{'reason/grading_metrics'}#video3#{'reason/grading_metrics'}
                ...
            4. **MUST** select **only the top 10** most relevant videos out of the provided 15.
            4. **MUST** include a clear and concise reason or grading metric for each video's ranking.
            5. **DO NOT** include any additional text, explanations, or deviations from the specified format.
            **Example Output**
            #video7228# Highly relevant: Comprehensive introduction to machine learning concepts aligning closely with the query. #video7035# Moderately relevant: Covers advanced deep learning topics but less aligned with introductory query. #video7143# Not relevant: Focuses on cooking recipes, unrelated to the query. ...
        """

    def __parse_response(self, response_text):
        parts = [p for p in response_text.split('#') if p.strip()]
        
        video_ids = [parts[i] for i in range(0, len(parts), 2)]
        return video_ids
    
    def rerank(self):
        self.__initialize_reranker()
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(self.prompt, safety_settings = safety_settings)
        # print(response.text)
        return self.__parse_response(response.text)