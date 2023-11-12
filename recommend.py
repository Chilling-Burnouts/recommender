from openai import OpenAI
import requests
import json
from operator import itemgetter
from flask import Flask, request, jsonify
import asyncio
from itertools import chain
from concurrent.futures import ThreadPoolExecutor
import numpy as np

app = Flask(__name__)

_search_API_key  = "pXStedvXkA9pMcNK1tWvx_4DesmTsIZ47qfTa6WkqFxgrCvCqJA0mpALQ53J"
_openai_key = "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"

def gpt_response(prompt, model="gpt-35-turbo"):
    client = OpenAI(api_key=_openai_key)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
        ],
        temperature=0.2
    )
    
    return completion.choices[0].message.content

def _get_similar_results(reference_company):
    """
    Parameters
    reference_company - dict with fields naics2022.primary.code, main_country and business_tags

    Returns
    json_response - response of search API request for similar companies
    """

    PAGE_SIZE = 50
    API_URL = f"https://data.veridion.com/search/v2/companies?page_size={PAGE_SIZE}"
    headers = {
        'x-api-key': _search_API_key,  # Replace with your actual API key
        'Content-type': 'application/json'
    }

    data = {
    "filters": {
        "and": [
            {
                "attribute": "company_naics_code",
                "relation": "equals",
                "value": reference_company['naics2022'],
                "strictness": 1
            },
            {
                "attribute": "company_location",
                "relation": "equals",
                "value": {
                    "country": reference_company['main_country'],
                }
            },
            {
                "attribute": "company_keywords",
                "relation": "match_expression",
                "value": {
                    "match": {
                        "operator": "or",
                        "operands": reference_company['business_tags']
                    }
                }
            }
        ]
    }
    }
    
    response = requests.post(API_URL, headers=headers, data=json.dumps(data))
    json_response = json.loads(response.text)
    return json_response

# response = gpt_response(f"I will give you a JSON describing a certain company. I want you to extract a list of 10 keywords that best describe this company. Do not focus on company name or branding, look for the domain it operates in, business specifications and other similar stuff. List them with commas only. The json starts here: \n \" {source_json} \"", model='gpt-4')
# response = response.split(',')


def _order_by_embeddings(results, reference):
    """
    Parameters
    results - array of long descriptions of each company
    reference - long description of reference company
    
    Returns
    sorted_indices - sorted indices of results, based on cosine similarity between reference and results' descriptions
    """
    client = OpenAI(api_key=_openai_key)
    
    embeddings = client.embeddings.create(input = results + [reference], model='text-embedding-ada-002').data
    # remove unimportant params from embeddings objects
    embeddings_results = [e.embedding for e in embeddings[:len(results)]]
    embeddings_reference = embeddings[-1].embedding
    embeddings_results_tensor = np.asarray(embeddings_results)
    embeddings_reference_tensor = np.asarray(embeddings_reference)

    similarities = np.matmul(embeddings_results_tensor, embeddings_reference_tensor)
    return np.argsort(-similarities), np.sort(similarities)    

# significant function to be exposed or whatever
def recommend_similar_companies(reference_company):
    similar_data = _get_similar_results(reference_company)
    
    print('sanki')
    list_of_descriptions = [data['long_description'] for data in similar_data['result']]
    sorted_indices, similarities = _order_by_embeddings(list_of_descriptions, reference_company['long_description'])

    sorted_list = list(itemgetter(*sorted_indices)(list_of_descriptions))
    # for company in sorted_list[:5]:
    #     print(company)
    #     print('-'*100)
    return sorted_list, similarities 

@app.route('/companies/recommended_portfolio', methods=['POST'])
async def recommend_for_portfolio():
    try:
        data = request.get_json()

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(recommend_similar_companies, data))
        # recommended_companies_list = recommend_similar_companies(data)
        all_results_texts = list(chain(*[result[0] for result in results]))
        all_results_sims = list(chain(*[result[1].tolist() for result in results]))
        sorted_indices = np.argsort(all_results_sims)
        sorted_list = list(itemgetter(*sorted_indices)(all_results_texts))

        for company in sorted_list[:5]:
            print(company)
            print('-'*100)
        return jsonify({'recommended_companies': sorted_list[:5]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/companies/recommended', methods=['POST'])
def recommended_companies():
    try:
        data = request.get_json()

        naics_primary_code = data.get('naics2022')
        main_country = data.get('main_country')
        business_tags = data.get('business_tags', [])
        long_description = data.get('long_description')

        recommended_companies_list = recommend_similar_companies(data)

        return jsonify({'recommended_companies': recommended_companies_list})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
# if __name__ == "__main__":
#     recommend_similar_companies({
#         'naics2022': {
#             'primary': {
#                 'code': '325413'
#             }
#         },
#         'main_country': "United States",
#         'keywords': ["Monoclonal Antibody", "Individuals With Disabilities", "Affirmative Action Policy", "FCOI Policy", "Pharmaceutical Company", "Project Funding"],
#         'long_description': "Mapp Biopharmaceutical is an American pharmaceutical company founded in 2003 by Larry Zeitlin and Kevin Whaley. Mapp Biopharmaceutical is based in San Diego, California. It is responsible for the research and development of ZMapp, a drug which is still under development and comprises three humanized monoclonal antibodies used as a treatment for Ebola virus disease. The drug was first tested in humans during the 2014 West Africa Ebola virus outbreak. The ZMapp drug is a result of the collaboration between Mapp Biopharmaceutical, LeafBio (the commercial arm of Mapp Biopharmaceutical), Defyrus Inc. (Toronto), the U.S. government, and the Public Health Agency of Canada. The antibody work came out of research projects funded by Defense Advanced Research Projects Agency (DARPA) more than a decade ago, and years of funding by the Public Health Agency of Canada. ZMapp is manufactured in the tobacco plant Nicotiana benthamiana in the bioproduction process known as \"pharming\" by Kentucky BioProcessing, a subsidiary of Reynolds American."
#     })