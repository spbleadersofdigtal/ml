from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import pickle
import os

model = None
data = None
index = None


def get_model():
    global model
    if not model:
        model = SentenceTransformer("sentence-transformers/LaBSE")
    return model


def get_data():
    global data
    if not data:
        with open("./data.pic", "rb") as file:
            data = pickle.load(file)
    for i in range(len(data)):
        if 'image' not in data[i].keys():
            data[i].update({'image': None})
    return data


def get_index():
    global index
    if not index:
        index = AnnoyIndex(768, "angular")
        index.load("index.ann")
    return index


def search(search_string):
    embs = get_model().encode([search_string])[0]
    indexes = get_index().get_nns_by_vector(embs, 5)
    res = []
    for i in indexes:
        try:
            res.append(get_data()[i])
        except: pass
    return list(map(lambda x: {'logo': x['image'], 'name': x['name'], 'description': x['description']}, res))


def search_investemnts(search_string):
    embs = get_model().encode([search_string])[0]
    indexes = get_index().get_nns_by_vector(embs, 5)
    res = []
    for i in indexes:
        try:
            res.append(get_data()[i])
        except:
            pass
    if not len(res):
        res = [{'total_investments': 2000000}]
    return list(map(lambda x: x['total_investments'], res))


comp_data = {
    "B2C": {
        "CAC": 10,
        "LTV": 12,
    },
    "B2B": {
        "CAC": 1000,
        "LTV": 1300
    },
    "TAM": {
        "AI": 40680000000000,
        "Business Software": 50600000000000,
        "IndustrialTech": 5000000000000,
        "E-commerce": 1440000000000000,
        "Advertising & Marketing": 47700000000000,
        "Hardware": 65700000000000,
        "RetailTech": 1080000000000,
        "ConstructionTech": 484200000000,
        "Web3": 5400000000000000,
        "EdTech": 10350000000000,
        "Business Intelligence": 2430000000000,
        "Cybersecurity": 15300000000000,
        "HrTech": 2160000000000,
        "Telecom & Communication": 162000000000000,
        "Media & Entertainment": 225000000000000,
        "FinTech": 810000000000000,
        "MedTech": 46080000000000,
        "Transport & Logistics": 90000000000000,
        "Gaming": 31230000000000,
        "FoodTech": 22230000000000,
        "WorkTech": 90000000000000,
        "Consumer Goods & Services": 8370000000000,
        "Aero & SpaceTech": 34200000000000,
        "Legal & RegTech": 720000000000,
        "Travel": 180000000000000,
        "PropTech": 2700000000000,
        "Energy": 162000000000,
        "GreenTech": 5580000000000,
    },
    "SAM": 0.3,
    "SOM": 0.13,
}




def calculate_metrics(category, description, type):
    cac = comp_data[type]['CAC']
    ltv = comp_data[type]['LTV']
    try:
        tam = comp_data['TAM'][category] // 1000
    except:
        tam = comp_data['TAM']['AI'] // 1000
    sam = comp_data['SAM'] * tam
    som = comp_data["SOM"] * comp_data['SAM'] * tam
    percent_now = 0.02
    percent_then = 0.18

    investments = search_investemnts(description)
    invest = sum(investments) / len(investments) / 300
    company_value = invest * 3
    invest_then = company_value * 1.3
    
    return [
        {
            'type': 'users_metrics',
            'value': {
                'cac': cac,
                'ltv': ltv
            },
        },
        {
            'type': 'market_values',
            'value': {
                'tam': tam,
                'sam': sam,
                'som': som
            }
        },
        {
            'type': 'percentage',
            'value': {
                'now': percent_now,
                'then': percent_then
            }
        },
        {
            'type': 'how_much_investments',
            'value': invest
        },
        {
            'type': 'company_value',
            'value': company_value
        },
        {
            'type': 'future_value',
            'value': invest_then
        }
    ]


def pdf_to_pptx(filename: str):
    os.system(f'pdf2pptx {filename}')